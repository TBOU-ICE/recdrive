"""Goal-free scene-router OPD with reverse-KL as the only training loss.

This trainer is intentionally additive.  It keeps the original scene-router
trainer untouched, but removes ExOPD target extrapolation and trajectory
regularisation for this experiment.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_dit_scene_router_distill_trainer import (
    ReCogDriveDiTSceneRouterDistillTrainer,
)


class ReCogDriveSceneRouterKDOnlyDistillTrainer(
    ReCogDriveDiTSceneRouterDistillTrainer
):
    """Student-on-policy, shared-variance reverse-KL for goal-free planners."""

    def __init__(
        self,
        *args,
        kd_weight: float = 1.0,
        kl_precision_clip: float = 25.0,
        collect_viz: bool = True,
        **kwargs,
    ):
        # These are invariants of this ablation, not user-tunable alternatives.
        kwargs["match_target"] = "mu"
        kwargs["exopd_lambda"] = 1.0
        kwargs["smooth_weight"] = 0.0
        super().__init__(*args, **kwargs)
        self.kd_weight = float(kd_weight)
        self.kl_precision_clip = float(kl_precision_clip)
        self.collect_viz = bool(collect_viz)
        self.last_viz = None
        if self.kd_weight <= 0.0:
            raise ValueError("kd_weight must be > 0")
        if self.kl_precision_clip <= 0.0:
            raise ValueError("kl_precision_clip must be > 0")

    def _normalised_precision(self, sigma2: torch.Tensor) -> torch.Tensor:
        """Stable KL precision, normalised so its maximum coefficient is one."""
        precision = (0.5 / sigma2.float().clamp(min=1e-6)).clamp(
            max=self.kl_precision_clip
        )
        return precision / self.kl_precision_clip

    @staticmethod
    def _shared_ddim_mean(
        planner,
        z_t: torch.Tensor,
        idx_batch: torch.Tensor,
        x0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Build a DDIM reverse mean using the same student sigma on both sides."""
        if planner.config.sampling_method != "ddim":
            raise NotImplementedError(
                "Scene-router KD-only reverse KL currently requires DDIM."
            )
        z_t = z_t.float()
        x0 = x0.float()
        sigma = sigma.float()
        alpha_t = planner.extract(planner.ddim_alphas, idx_batch, z_t.shape).float()
        alpha_prev = planner.extract(
            planner.ddim_alphas_prev, idx_batch, z_t.shape
        ).float()
        sqrt_one_minus_alpha_t = planner.extract(
            planner.ddim_sqrt_one_minus_alphas, idx_batch, z_t.shape
        ).float().clamp(min=1e-8)
        pred_noise = (z_t - alpha_t.sqrt() * x0) / sqrt_one_minus_alpha_t
        pred_direction = (
            1.0 - alpha_prev - sigma.pow(2)
        ).clamp(min=0.0).sqrt() * pred_noise
        return alpha_prev.sqrt() * x0 + pred_direction

    @staticmethod
    def _trajectory_gap_m(
        planner, trajectory_a: torch.Tensor, trajectory_b: torch.Tensor
    ) -> torch.Tensor:
        a = planner.denorm_odo(trajectory_a.float())
        b = planner.denorm_odo(trajectory_b.float())
        return (a[..., :2] - b[..., :2]).norm(dim=-1).mean()

    def compute_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        ref_planner: Optional[torch.nn.Module],
        vl_features: torch.Tensor,
        action_input,
        bucket_per_sample: List[str],
    ) -> BatchFeature:
        if ref_planner is not None:
            raise ValueError("KD-only scene-router OPD must not build an ExOPD reference.")
        if hasattr(student_planner, "predict_goal_from_encoded"):
            raise TypeError("KD-only scene-router OPD requires a goal-free student.")
        if any(hasattr(teacher, "goal_context") for teacher in teacher_planners.values()):
            raise TypeError("This experiment requires the original goal-free RL teachers.")

        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        history = action_input.his_traj
        ego_status = action_input.status_feature
        gt_trajectory = action_input.action.float()

        # Reverse KL is an expectation over student states.  Roll out the current
        # student stochastically, then detach those sampled states.
        with torch.no_grad():
            vl_s0, history_s0, ego_s0 = self._encode(
                student_planner,
                vl_features,
                history,
                ego_status,
                student_dtype,
            )
            chain = self._sample_chain(
                student_planner,
                vl_s0,
                history_s0,
                ego_s0,
                batch_size,
                device,
                student_dtype,
            )
        num_steps = chain.shape[1] - 1
        ddim_timesteps = [
            int(student_planner.ddim_t[index].item())
            for index in range(num_steps)
        ]

        teacher_encodings: Dict[str, tuple] = {}
        with torch.no_grad():
            for name, teacher in teacher_planners.items():
                teacher.eval()
                teacher_encodings[name] = self._encode(
                    teacher, vl_features, history, ego_status, torch.float32
                )

        # Recompute the student branch with gradients at the detached states.
        vl_s, history_s, ego_s = self._encode(
            student_planner,
            vl_features,
            history,
            ego_status,
            student_dtype,
        )

        resolved_buckets = [self._resolve_bucket(name) for name in bucket_per_sample]
        bucket_to_indices: Dict[str, List[int]] = {}
        for sample_index, bucket in enumerate(resolved_buckets):
            bucket_to_indices.setdefault(bucket, []).append(sample_index)

        total_kd = vl_features.new_zeros((), dtype=torch.float32)
        max_sample_kd = vl_features.new_zeros((), dtype=torch.float32)
        step_kls: List[torch.Tensor] = []
        step_sigmas: List[torch.Tensor] = []
        step_x0_gaps: List[torch.Tensor] = []
        step_metrics: Dict[str, torch.Tensor] = {}
        bucket_kls: Dict[str, List[torch.Tensor]] = {
            bucket: [] for bucket in self.bucket_names
        }
        final_student_x0 = None
        final_teacher_x0 = torch.zeros(
            batch_size, chain.shape[2], chain.shape[3], device=device
        )
        viz_steps = {0, num_steps // 2, num_steps - 1}
        viz_step_predictions = []

        for step in range(num_steps):
            z_t = chain[:, step].to(student_dtype)
            t_batch = student_planner.make_timesteps(
                batch_size, ddim_timesteps[step], device
            )
            idx_batch = student_planner.make_timesteps(batch_size, step, device)

            _, student_logvar, student_x0 = student_planner.p_mean_variance(
                z_t,
                t_batch,
                idx_batch,
                vl_s,
                history_s,
                ego_s,
                deterministic=False,
            )
            # The variance is fixed sampling geometry, not an optimisation escape
            # route.  Teacher and student means below both use this same value.
            student_sigma = (
                0.5 * student_logvar.float().clamp(-40.0, 20.0)
            ).exp().detach()
            weighting_sigma = student_sigma.clamp(min=self.min_sigma)
            precision = self._normalised_precision(weighting_sigma.pow(2))
            student_mean = self._shared_ddim_mean(
                student_planner, z_t, idx_batch, student_x0, student_sigma
            )

            step_kd_sum = vl_features.new_zeros((), dtype=torch.float32)
            step_gap_sum = vl_features.new_zeros((), dtype=torch.float32)
            samples_used = 0
            teacher_x0_full = torch.zeros_like(final_teacher_x0)

            for bucket, indices in bucket_to_indices.items():
                selection = torch.as_tensor(indices, device=device, dtype=torch.long)
                teacher = teacher_planners[bucket]
                encoding = teacher_encodings[bucket]
                with torch.no_grad():
                    # deterministic only changes the native teacher mean/sigma;
                    # x0 itself is unchanged.  We explicitly reconstruct the
                    # teacher mean below with the student's shared sigma.
                    _, _, teacher_x0 = teacher.p_mean_variance(
                        z_t[selection].float(),
                        t_batch[selection],
                        idx_batch[selection],
                        encoding[0][selection],
                        encoding[1][selection],
                        encoding[2][selection],
                        deterministic=True,
                    )
                    teacher_mean = self._shared_ddim_mean(
                        student_planner,
                        z_t[selection],
                        idx_batch[selection],
                        teacher_x0,
                        student_sigma[selection],
                    )

                # Per-coordinate regularised shared-covariance reverse KL.
                kl_per_sample = (
                    (student_mean[selection] - teacher_mean).pow(2)
                    * precision[selection]
                ).mean(dim=(1, 2))
                step_kd_sum = step_kd_sum + kl_per_sample.sum()
                max_sample_kd = torch.maximum(
                    max_sample_kd, kl_per_sample.detach().max()
                )
                bucket_kls[bucket].append(kl_per_sample.mean().detach())

                with torch.no_grad():
                    gap = self._trajectory_gap_m(
                        student_planner, student_x0[selection], teacher_x0
                    )
                    step_gap_sum = step_gap_sum + gap * len(indices)
                    teacher_x0_full[selection] = teacher_x0.float()
                samples_used += len(indices)

            step_kd = step_kd_sum / max(samples_used, 1)
            step_gap = step_gap_sum / max(samples_used, 1)
            total_kd = total_kd + step_kd
            step_kls.append(step_kd.detach())
            step_x0_gaps.append(step_gap.detach())
            step_sigmas.append(student_sigma.float().mean().detach())
            final_student_x0 = student_x0

            with torch.no_grad():
                student_trajectory = student_planner.denorm_odo(student_x0.float())
                teacher_trajectory = student_planner.denorm_odo(teacher_x0_full)
                student_error = (
                    student_trajectory[..., :2] - gt_trajectory[..., :2]
                ).norm(dim=-1)
                teacher_error = (
                    teacher_trajectory[..., :2] - gt_trajectory[..., :2]
                ).norm(dim=-1)
                step_metrics[f"fde_gt_student_step_{step}_m"] = (
                    student_error[:, -1].mean().detach()
                )
                step_metrics[f"fde_gt_student_step_{step}_ade_m"] = (
                    student_error.mean().detach()
                )
                step_metrics[f"fde_gt_teacher_step_{step}_m"] = (
                    teacher_error[:, -1].mean().detach()
                )
                step_metrics[f"fde_gt_teacher_step_{step}_ade_m"] = (
                    teacher_error.mean().detach()
                )
                if self.collect_viz and step in viz_steps:
                    viz_step_predictions.append(
                        {
                            "step": step,
                            "timestep": ddim_timesteps[step],
                            "student_x0": student_trajectory.detach().cpu(),
                            "teacher_x0": teacher_trajectory.detach().cpu(),
                        }
                    )

            if step == num_steps - 1:
                final_teacher_x0 = teacher_x0_full

        reverse_kl = total_kd / max(num_steps, 1)
        loss = self.kd_weight * reverse_kl

        # EtaFixed carries an inf-valued, frozen eta_logit in some checkpoints.
        # Keep a finite zero graph edge for DDP without introducing another loss.
        if hasattr(student_planner, "eta") and hasattr(
            student_planner.eta, "eta_logit"
        ):
            eta_logit = student_planner.eta.eta_logit
            loss = loss + torch.nan_to_num(
                eta_logit, nan=0.0, posinf=0.0, neginf=0.0
            ).sum() * 0.0
        if not torch.isfinite(loss):
            # Numerical safety only: this returns zero and supplies no anchor or
            # trajectory supervision.
            loss = self._finite_anchor_loss(student_planner)

        student_trajectory = student_planner.denorm_odo(final_student_x0.float())
        with torch.no_grad():
            teacher_trajectory = student_planner.denorm_odo(final_teacher_x0)
            student_error = (
                student_trajectory[..., :2] - gt_trajectory[..., :2]
            ).norm(dim=-1)
            teacher_error = (
                teacher_trajectory[..., :2] - gt_trajectory[..., :2]
            ).norm(dim=-1)

        data = {
            "loss": loss,
            "distill_loss": reverse_kl.detach(),
            "reverse_kl_loss": reverse_kl.detach(),
            "student_fde_gt_m": student_error[:, -1].mean().detach(),
            "student_ade_gt_m": student_error.mean().detach(),
            "fde_gt_teacher_final_m": teacher_error[:, -1].mean().detach(),
            "fde_gt_teacher_ade_m": teacher_error.mean().detach(),
            "x0_gap_m": torch.stack(step_x0_gaps).mean(),
            "gauss_kl_mean": torch.stack(step_kls).mean(),
            "kl_ungated_sample_max": max_sample_kd,
            "sigma_mean": torch.stack(step_sigmas).mean(),
            "chain_abs_max": chain.float().abs().max().detach(),
            "denoising_steps": torch.tensor(float(num_steps), device=device),
            "exopd_lambda": torch.tensor(1.0, device=device),
        }
        data.update(step_metrics)

        kl_sum = torch.stack(step_kls).sum().clamp_min(1e-12)
        for step, step_kl in enumerate(step_kls):
            data[f"kl_step_{step}"] = step_kl
            data[f"kl_step_{step}_fraction"] = step_kl / kl_sum

        zero = torch.tensor(0.0, device=device)
        for bucket in self.bucket_names:
            values = bucket_kls[bucket]
            data[f"kl_{bucket}_mean"] = (
                torch.stack(values).mean() if values else zero
            )
            data[f"n_samples_{bucket}"] = torch.tensor(
                float(len(bucket_to_indices.get(bucket, []))), device=device
            )

        if self.collect_viz:
            self.last_viz = {
                "gt_traj": gt_trajectory.detach().cpu(),
                "student_traj": student_trajectory.detach().float().cpu(),
                "teacher_traj": teacher_trajectory.detach().float().cpu(),
                "step_predictions": viz_step_predictions,
                "buckets": list(resolved_buckets),
            }

        return BatchFeature(data=data)

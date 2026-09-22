"""Goal-free-student, privileged-teacher KD-only OPD with step metrics.

The student is a plain ``ReCogDriveDiffusionPlanner``: it has no goal
predictor and receives no goal injection.  The routed teachers retain their GT
endpoint conditioning.  This is isolated from every existing trainer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_dit_scene_router_distill_trainer import (
    ReCogDriveDiTSceneRouterDistillTrainer,
)


class ReCogDriveGoalFreeKDOnlyDistillTrainer(
    ReCogDriveDiTSceneRouterDistillTrainer
):
    """Clipped-precision shared-DDIM mean matching (KL surrogate)."""

    def __init__(
        self,
        *args,
        kd_weight: float = 1.0,
        kl_precision_clip: float = 25.0,
        collect_viz: bool = True,
        rollout_min_sigma: Optional[float] = None,
        rollout_noise_clip: float = 5.0,
        rollout_deterministic: bool = False,
        diagnostic_gradients: bool = False,
        fail_on_nonfinite: bool = False,
        **kwargs,
    ):
        # This ablation always matches shared-variance DDIM reverse means and
        # does not use ExOPD extrapolation or trajectory regularizers.
        kwargs["match_target"] = "mu"
        kwargs["exopd_lambda"] = 1.0
        kwargs["smooth_weight"] = 0.0
        super().__init__(*args, **kwargs)
        self.kd_weight = float(kd_weight)
        self.kl_precision_clip = float(kl_precision_clip)
        self.collect_viz = bool(collect_viz)
        self.last_viz = None
        # None preserves existing launchers; new correction pipeline sets this
        # independently of the variance floor used in the loss precision.
        self._legacy_rollout_sigma = rollout_min_sigma is None
        self.rollout_min_sigma = self.min_sigma if rollout_min_sigma is None else float(rollout_min_sigma)
        self.rollout_noise_clip = float(rollout_noise_clip)
        self.rollout_deterministic = bool(rollout_deterministic)
        self.diagnostic_gradients = bool(diagnostic_gradients)
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        if self.rollout_min_sigma < 0 or self.rollout_noise_clip <= 0:
            raise ValueError("Invalid rollout noise settings")

    def _sample_chain(self, planner, vl, his, ego, batch_size, device, dtype):
        z = torch.randn((batch_size, planner.config.action_horizon, planner.config.action_dim), device=device, dtype=dtype)
        chain = [z.clone()]
        for i in range(planner.ddim_steps):
            t = planner.make_timesteps(batch_size, int(planner.ddim_t[i]), device)
            idx = planner.make_timesteps(batch_size, i, device)
            mu, logvar, _ = planner.p_mean_variance(z, t, idx, vl, his, ego, deterministic=self.rollout_deterministic)
            sigma = (
                self._safe_sigma(logvar, self.rollout_min_sigma, dtype)
                if self._legacy_rollout_sigma
                else (0.5 * logvar).exp().to(dtype).clamp_min(self.rollout_min_sigma)
            )
            if self.rollout_deterministic:
                sigma = torch.zeros_like(sigma)
            noise = torch.randn_like(z).clamp(-self.rollout_noise_clip, self.rollout_noise_clip)
            z = (mu + sigma * noise).detach()
            chain.append(z.clone())
        return torch.stack(chain, dim=1)

    def _precision(self, sigma2: torch.Tensor) -> torch.Tensor:
        if self.kl_precision_clip <= 0.0:
            raise ValueError("kl_precision_clip must be > 0")
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
        """Reconstruct a reverse mean with one shared student DDIM sigma."""
        if planner.config.sampling_method != "ddim":
            raise NotImplementedError(
                "Goal-free KD-only shared reverse KL requires DDIM."
            )
        z = z_t.float()
        x0 = x0.float()
        sigma = sigma.float()
        alpha_t = planner.extract(
            planner.ddim_alphas, idx_batch, z.shape
        ).float()
        alpha_prev = planner.extract(
            planner.ddim_alphas_prev, idx_batch, z.shape
        ).float()
        sqrt_one_minus_alpha_t = planner.extract(
            planner.ddim_sqrt_one_minus_alphas, idx_batch, z.shape
        ).float().clamp(min=1e-8)
        pred_noise = (z - alpha_t.sqrt() * x0) / sqrt_one_minus_alpha_t
        pred_dir = (
            1.0 - alpha_prev - sigma.pow(2)
        ).clamp(min=0.0).sqrt() * pred_noise
        return alpha_prev.sqrt() * x0 + pred_dir

    @staticmethod
    def _waypoint_gap_m(
        planner, x0_a: torch.Tensor, x0_b: torch.Tensor
    ) -> torch.Tensor:
        a = planner.denorm_odo(x0_a.float())
        b = planner.denorm_odo(x0_b.float())
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
        del ref_planner
        if hasattr(student_planner, "predict_goal_from_encoded"):
            raise TypeError(
                "Goal-free KD-only received a predicted-goal student; "
                "use a plain ReCogDriveDiffusionPlanner."
            )

        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature
        gt_traj = action_input.action.float()
        gt_goal = gt_traj[:, -1, :].contiguous()

        # Reverse-KL state expectation comes from a detached, goal-free student
        # rollout.  No goal is computed or bound anywhere on the student side.
        with torch.no_grad():
            vl_s0, his_s0, ego_s0 = self._encode(
                student_planner,
                vl_features,
                his_traj,
                ego_status,
                student_dtype,
            )
            chain = self._sample_chain(
                student_planner,
                vl_s0,
                his_s0,
                ego_s0,
                batch_size,
                device,
                student_dtype,
            )
        num_steps = chain.shape[1] - 1
        ddim_t_list = [
            int(student_planner.ddim_t[i].item()) for i in range(num_steps)
        ]

        teacher_encodings: Dict[str, tuple] = {}
        with torch.no_grad():
            for name, teacher in teacher_planners.items():
                teacher.eval()
                teacher_encodings[name] = self._encode(
                    teacher, vl_features, his_traj, ego_status, torch.float32
                )

        # Recompute only the student prediction at each sampled state with grad.
        vl_s, his_s, ego_s = self._encode(
            student_planner,
            vl_features,
            his_traj,
            ego_status,
            student_dtype,
        )
        resolved = [self._resolve_bucket(bucket) for bucket in bucket_per_sample]
        bucket_to_indices: Dict[str, List[int]] = {}
        for index, bucket in enumerate(resolved):
            bucket_to_indices.setdefault(bucket, []).append(index)

        total_kd = vl_features.new_zeros((), dtype=torch.float32)
        kd_sample_max = vl_features.new_zeros((), dtype=torch.float32)
        kl_steps: List[torch.Tensor] = []
        sigma_steps: List[torch.Tensor] = []
        x0_gap_steps: List[torch.Tensor] = []
        step_metrics: Dict[str, torch.Tensor] = {}
        per_bucket_kl: Dict[str, List[torch.Tensor]] = {
            bucket: [] for bucket in self.bucket_names
        }
        last_student_x0 = None
        last_teacher_x0_full = torch.zeros(
            batch_size, chain.shape[2], chain.shape[3], device=device
        )
        viz_steps = {0, num_steps // 2, num_steps - 1}
        viz_step_predictions = []

        for step in range(num_steps):
            z_t = chain[:, step].to(student_dtype)
            t_batch = student_planner.make_timesteps(
                batch_size, ddim_t_list[step], device
            )
            idx_batch = student_planner.make_timesteps(batch_size, step, device)

            _, logvar_s, x0_s = student_planner.p_mean_variance(
                z_t,
                t_batch,
                idx_batch,
                vl_s,
                his_s,
                ego_s,
                deterministic=self.rollout_deterministic,
            )
            sigma_geom = (
                0.5 * logvar_s.float().clamp(-40.0, 20.0)
            ).exp().detach()
            sigma_weight = sigma_geom.clamp(min=self.min_sigma)
            precision = self._precision(
                sigma_weight.pow(2).clamp(min=1e-6)
            )
            mu_s_shared = self._shared_ddim_mean(
                student_planner, z_t, idx_batch, x0_s, sigma_geom
            )

            step_kd_sum = vl_features.new_zeros((), dtype=torch.float32)
            step_gap_sum = vl_features.new_zeros((), dtype=torch.float32)
            n_used = 0
            teacher_x0_full = torch.zeros_like(last_teacher_x0_full)

            for bucket, indices in bucket_to_indices.items():
                sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                teacher = teacher_planners[bucket]
                enc = teacher_encodings[bucket]
                with torch.no_grad():
                    # Only the frozen teacher receives privileged information.
                    with teacher.goal_context(gt_goal[sel]):
                        _, _, x0_t = teacher.p_mean_variance(
                            z_t[sel].float(),
                            t_batch[sel],
                            idx_batch[sel],
                            enc[0][sel],
                            enc[1][sel],
                            enc[2][sel],
                            deterministic=True,
                        )
                    mu_t_shared = self._shared_ddim_mean(
                        student_planner,
                        z_t[sel],
                        idx_batch[sel],
                        x0_t,
                        sigma_geom[sel],
                    )

                kl_per = (
                    (mu_s_shared[sel] - mu_t_shared).pow(2) * precision[sel]
                ).mean(dim=(1, 2))
                step_kd_sum = step_kd_sum + kl_per.sum()
                kd_sample_max = torch.maximum(
                    kd_sample_max, kl_per.detach().max()
                )
                per_bucket_kl[bucket].append(kl_per.mean().detach())

                with torch.no_grad():
                    gap = self._waypoint_gap_m(
                        student_planner, x0_s[sel], x0_t
                    )
                    step_gap_sum = step_gap_sum + gap * len(indices)
                    teacher_x0_full[sel] = x0_t.float()
                n_used += len(indices)

            step_kd = step_kd_sum / max(n_used, 1)
            step_gap = step_gap_sum / max(n_used, 1)
            if self.diagnostic_gradients and torch.is_grad_enabled():
                gx = torch.autograd.grad(self.kd_weight * step_kd / num_steps, x0_s, retain_graph=True)[0]
                step_metrics[f"kd_grad_x0_step_{step}"] = gx.detach().norm()
            total_kd = total_kd + step_kd
            kl_steps.append(step_kd.detach())
            x0_gap_steps.append(step_gap.detach())
            sigma_steps.append(sigma_geom.float().mean().detach())
            last_student_x0 = x0_s

            with torch.no_grad():
                student_step_traj = student_planner.denorm_odo(x0_s.float())
                teacher_step_traj = student_planner.denorm_odo(teacher_x0_full)
                student_step_err = (
                    student_step_traj[..., :2] - gt_traj[..., :2]
                ).norm(dim=-1)
                teacher_step_err = (
                    teacher_step_traj[..., :2] - gt_traj[..., :2]
                ).norm(dim=-1)
                step_metrics[f"fde_gt_student_step_{step}_m"] = (
                    student_step_err[:, -1].mean().detach()
                )
                step_metrics[f"fde_gt_student_step_{step}_ade_m"] = (
                    student_step_err.mean().detach()
                )
                step_metrics[f"fde_gt_teacher_step_{step}_m"] = (
                    teacher_step_err[:, -1].mean().detach()
                )
                step_metrics[f"fde_gt_teacher_step_{step}_ade_m"] = (
                    teacher_step_err.mean().detach()
                )
                if self.collect_viz and step in viz_steps:
                    viz_step_predictions.append(
                        {
                            "step": step,
                            "timestep": ddim_t_list[step],
                            "student_x0": student_step_traj.detach().cpu(),
                            "teacher_x0": teacher_step_traj.detach().cpu(),
                        }
                    )

            if step == num_steps - 1:
                last_teacher_x0_full = teacher_x0_full

        kd_loss = total_kd / max(num_steps, 1)
        loss = self.kd_weight * kd_loss
        if hasattr(student_planner, "eta") and hasattr(
            student_planner.eta, "eta_logit"
        ):
            eta_logit = student_planner.eta.eta_logit
            loss = loss + torch.nan_to_num(
                eta_logit, nan=0.0, posinf=0.0, neginf=0.0
            ).sum() * 0.0
        if not torch.isfinite(loss):
            if self.fail_on_nonfinite:
                raise FloatingPointError("Non-finite OPD loss")
            loss = self._finite_anchor_loss(student_planner)

        pred_traj_s = student_planner.denorm_odo(last_student_x0.float())
        with torch.no_grad():
            teacher_traj_final = student_planner.denorm_odo(
                last_teacher_x0_full
            )
            student_err = (
                pred_traj_s[..., :2] - gt_traj[..., :2]
            ).norm(dim=-1)
            teacher_err = (
                teacher_traj_final[..., :2] - gt_traj[..., :2]
            ).norm(dim=-1)

        data = {
            "loss": loss,
            "distill_loss": kd_loss.detach(),
            "reverse_kl_loss": kd_loss.detach(),  # legacy metric name; surrogate, not exact KL
            "kl_proxy_loss": kd_loss.detach(),
            "student_fde_gt_m": student_err[:, -1].mean().detach(),
            "student_ade_gt_m": student_err.mean().detach(),
            "fde_gt_teacher_final_m": teacher_err[:, -1].mean().detach(),
            "fde_gt_teacher_ade_m": teacher_err.mean().detach(),
            "x0_gap_m": torch.stack(x0_gap_steps).mean(),
            "gauss_kl_mean": torch.stack(kl_steps).mean(),
            "kl_ungated_sample_max": kd_sample_max,
            "sigma_mean": torch.stack(sigma_steps).mean(),
            "chain_abs_max": chain.float().abs().max().detach(),
            "denoising_steps": torch.tensor(float(num_steps), device=device),
        }
        data.update(step_metrics)

        kl_sum = torch.stack(kl_steps).sum().clamp_min(1e-12)
        for step, step_kl in enumerate(kl_steps):
            data[f"kl_step_{step}"] = step_kl
            data[f"kl_step_{step}_fraction"] = step_kl / kl_sum

        zero = torch.tensor(0.0, device=device)
        for bucket in self.bucket_names:
            values = per_bucket_kl[bucket]
            data[f"kl_{bucket}_mean"] = (
                torch.stack(values).mean() if values else zero
            )
            data[f"n_samples_{bucket}"] = torch.tensor(
                float(len(bucket_to_indices.get(bucket, []))), device=device
            )

        if self.collect_viz:
            self.last_viz = {
                "gt_traj": gt_traj.detach().cpu(),
                "goal": gt_goal.detach().cpu(),
                "student_traj": pred_traj_s.detach().float().cpu(),
                "teacher_traj": teacher_traj_final.detach().float().cpu(),
                "step_predictions": viz_step_predictions,
                "buckets": list(resolved),
            }

        return BatchFeature(data=data)

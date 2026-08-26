"""GoalBridge OPD: predicted-goal student -> privileged routed teachers.

Key changes versus the old privileged-goal OPD:
1) the student predicts a deployable goal from observation features;
2) the student DDIM chain and all student denoising calls are conditioned on
   that predicted goal;
3) distillation matches DDIM reverse-transition means (Gaussian KL surrogate)
   with clipped/normalised precision rather than the old unbounded x0 loss;
4) a recoverability score gates privileged KD and a frozen goal-free anchor;
5) a direct goal regression loss explicitly teaches the missing information.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_dit_scene_router_distill_trainer import (
    ReCogDriveDiTSceneRouterDistillTrainer,
)

_LOG_2PI_E = math.log(2.0 * math.pi * math.e)


class ReCogDriveGoalBridgeDistillTrainer(ReCogDriveDiTSceneRouterDistillTrainer):
    """Recoverability-aware goal-conditioned multi-teacher OPD."""

    def __init__(
        self,
        *args,
        goal_loss_weight: float = 1.0,
        kd_weight: float = 1.0,
        anchor_weight: float = 0.15,
        recoverability_tau_m: float = 2.0,
        recoverability_floor: float = 0.05,
        kl_precision_clip: float = 25.0,
        collect_viz: bool = False,
        **kwargs,
    ):
        # This trainer always distils reverse means, i.e. the shared-variance
        # Gaussian reverse-transition KL surrogate.
        kwargs["match_target"] = "mu"
        kwargs["exopd_lambda"] = 1.0
        super().__init__(*args, **kwargs)
        self.goal_loss_weight = float(goal_loss_weight)
        self.kd_weight = float(kd_weight)
        self.anchor_weight = float(anchor_weight)
        self.recoverability_tau_m = float(recoverability_tau_m)
        self.recoverability_floor = float(recoverability_floor)
        self.kl_precision_clip = float(kl_precision_clip)
        self.collect_viz = bool(collect_viz)
        self.last_viz = None

    def _precision(self, sigma2: torch.Tensor) -> torch.Tensor:
        """Clipped shared-variance KL precision, rescaled to ``(0, 1]``.

        Dividing by the fixed clip value only changes the global KD scale (which is
        already controlled by ``kd_weight``), while preserving relative weighting
        across DDIM steps.  The previous per-step mean-normalisation accidentally
        cancelled almost all timestep-dependent precision information.
        """
        if self.kl_precision_clip <= 0.0:
            raise ValueError("kl_precision_clip must be > 0")
        precision = (0.5 / sigma2.float().clamp(min=1e-6)).clamp(max=self.kl_precision_clip)
        return precision / self.kl_precision_clip

    @staticmethod
    def _shared_ddim_mean(
        planner,
        z_t: torch.Tensor,
        idx_batch: torch.Tensor,
        x0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a DDIM reverse mean using a caller-supplied variance.

        This is the important detail for a valid shared-variance Gaussian KL.
        ``p_mean_variance(..., deterministic=False)`` and
        ``p_mean_variance(..., deterministic=True)`` otherwise use different eta
        values, so directly comparing their returned means is not a shared-variance
        KL.  We instead use the same detached ``sigma`` for student, teacher and
        anchor, while keeping each model's own predicted ``x0``.
        """
        if planner.config.sampling_method != "ddim":
            raise NotImplementedError("GoalBridge shared reverse KL currently requires DDIM.")
        z = z_t.float()
        x0 = x0.float()
        sigma = sigma.float()
        alpha_t = planner.extract(planner.ddim_alphas, idx_batch, z.shape).float()
        alpha_prev = planner.extract(planner.ddim_alphas_prev, idx_batch, z.shape).float()
        sqrt_one_minus_alpha_t = planner.extract(
            planner.ddim_sqrt_one_minus_alphas, idx_batch, z.shape
        ).float().clamp(min=1e-8)
        pred_noise = (z - alpha_t.sqrt() * x0) / sqrt_one_minus_alpha_t
        pred_dir = (1.0 - alpha_prev - sigma.pow(2)).clamp(min=0.0).sqrt() * pred_noise
        return alpha_prev.sqrt() * x0 + pred_dir

    def compute_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        ref_planner: Optional[torch.nn.Module],  # unused; kept for caller compatibility
        vl_features: torch.Tensor,
        action_input,
        bucket_per_sample: List[str],
        anchor_planner: Optional[torch.nn.Module] = None,
    ) -> BatchFeature:
        if not hasattr(student_planner, "predict_goal_from_encoded"):
            raise TypeError("GoalBridge requires PredictedGoalDiffusionPlanner as the student.")

        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature
        gt_traj = action_input.action.float()
        gt_goal = gt_traj[:, -1, :].contiguous()

        # Student encodings with gradient + explicit goal recovery.
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, student_dtype
        )
        pred_goal, pred_goal_norm = student_planner.predict_goal_from_encoded(vl_s, his_s, ego_s)
        gt_goal_norm = student_planner.norm_odo(gt_goal.unsqueeze(1)).squeeze(1).to(pred_goal_norm.dtype)
        # Current privileged teachers use goal_use_heading=False, so do not spend
        # auxiliary capacity fitting a heading channel that the goal conditioner
        # ignores.  If heading conditioning is enabled later, train all 3 dims.
        use_heading = bool(getattr(getattr(student_planner, "goal_encoder", None), "use_heading", False))
        goal_dims = 3 if use_heading else 2
        goal_loss = F.smooth_l1_loss(
            pred_goal_norm[..., :goal_dims], gt_goal_norm[..., :goal_dims], reduction="mean"
        )

        # Recoverability is deliberately detached: it is a curriculum/gating
        # signal, not something the goal head should be able to game.
        goal_err_xy = (pred_goal.float()[..., :2] - gt_goal[..., :2]).norm(dim=-1)
        tau = max(self.recoverability_tau_m, 1e-3)
        recoverability = torch.exp(-(goal_err_xy.detach() ** 2) / (2.0 * tau * tau))
        recoverability = recoverability.clamp(min=self.recoverability_floor, max=1.0)

        # Detached on-policy student chain, already conditioned on the student's
        # OWN predicted goal (never on GT goal).
        with torch.no_grad():
            with student_planner.goal_context(pred_goal.detach()):
                chain = self._sample_chain(
                    student_planner,
                    vl_s.detach(),
                    his_s.detach(),
                    ego_s.detach(),
                    batch_size,
                    device,
                    student_dtype,
                )
        num_steps = chain.shape[1] - 1
        ddim_t_list = [int(student_planner.ddim_t[i].item()) for i in range(num_steps)]

        # Frozen teacher and anchor encodings.
        teacher_encodings: Dict[str, tuple] = {}
        with torch.no_grad():
            for name, teacher in teacher_planners.items():
                teacher.eval()
                teacher_encodings[name] = self._encode(
                    teacher, vl_features, his_traj, ego_status, torch.float32
                )
            anchor_encoding = (
                self._encode(anchor_planner, vl_features, his_traj, ego_status, torch.float32)
                if anchor_planner is not None
                else None
            )

        resolved = [self._resolve_bucket(b) for b in bucket_per_sample]
        bucket_to_indices: Dict[str, List[int]] = {}
        for i, bucket in enumerate(resolved):
            bucket_to_indices.setdefault(bucket, []).append(i)

        total_kd = vl_features.new_zeros((), dtype=torch.float32)
        total_anchor = vl_features.new_zeros((), dtype=torch.float32)
        sigma_list: List[torch.Tensor] = []
        entropy_steps: List[torch.Tensor] = []
        x0_gap_steps: List[torch.Tensor] = []
        kl_steps: List[torch.Tensor] = []
        last_student_x0 = None
        last_teacher_x0_full = torch.zeros(batch_size, chain.shape[2], chain.shape[3], device=device)
        per_bucket_kl: Dict[str, List[torch.Tensor]] = {b: [] for b in self.bucket_names}

        for step in range(num_steps):
            z_t = chain[:, step].to(student_dtype)
            t_batch = student_planner.make_timesteps(batch_size, ddim_t_list[step], device)
            idx_batch = student_planner.make_timesteps(batch_size, step, device)

            with student_planner.goal_context(pred_goal):
                _, logvar_s, x0_s = student_planner.p_mean_variance(
                    z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False
                )
            # Use the student's *actual* DDIM sigma to reconstruct both reverse
            # means.  The min_sigma floor is only a numerical floor for the KL
            # denominator; using that floor inside the DDIM geometry can be invalid
            # at late steps where 1-alpha_prev < min_sigma^2.
            sigma_geom = (0.5 * logvar_s.float().clamp(-40.0, 20.0)).exp().detach()
            sigma_weight = sigma_geom.clamp(min=self.min_sigma)
            sigma2 = sigma_weight.pow(2).clamp(min=1e-6)
            precision = self._precision(sigma2)
            mu_s_shared = self._shared_ddim_mean(
                student_planner, z_t, idx_batch, x0_s, sigma_geom
            )
            sigma_list.append(sigma_geom.float().mean().detach())

            traj_dims = float(x0_s.shape[1] * x0_s.shape[2])
            entropy_steps.append(
                ((0.5 * (_LOG_2PI_E + sigma2.log())).mean(dim=(1, 2)) * traj_dims)
                .mean()
                .detach()
            )

            with torch.no_grad():
                if anchor_planner is not None:
                    _, _, x0_anchor = anchor_planner.p_mean_variance(
                        z_t.float(), t_batch, idx_batch,
                        anchor_encoding[0], anchor_encoding[1], anchor_encoding[2],
                        deterministic=True,
                    )
                    mu_anchor = self._shared_ddim_mean(
                        student_planner, z_t, idx_batch, x0_anchor, sigma_geom
                    )
                else:
                    mu_anchor = None

            step_kd_sum = vl_features.new_zeros((), dtype=torch.float32)
            step_anchor_sum = vl_features.new_zeros((), dtype=torch.float32)
            step_gap_sum = vl_features.new_zeros((), dtype=torch.float32)
            n_used = 0

            for bucket, indices in bucket_to_indices.items():
                sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                teacher = teacher_planners[bucket]
                enc = teacher_encodings[bucket]
                with torch.no_grad():
                    with teacher.goal_context(gt_goal[sel]):
                        _, _, x0_t = teacher.p_mean_variance(
                            z_t[sel].float(), t_batch[sel], idx_batch[sel],
                            enc[0][sel], enc[1][sel], enc[2][sel],
                            deterministic=True,
                        )
                    mu_t_shared = self._shared_ddim_mean(
                        student_planner, z_t[sel], idx_batch[sel], x0_t, sigma_geom[sel]
                    )

                # Shared-variance reverse-transition Gaussian KL mean term. Mean
                # over trajectory dims keeps scale independent of horizon length.
                kl_per = (
                    (mu_s_shared[sel] - mu_t_shared).pow(2) * precision[sel]
                ).mean(dim=(1, 2))
                c = recoverability[sel]
                step_kd_sum = step_kd_sum + (c * kl_per).sum()
                per_bucket_kl[bucket].append(kl_per.mean().detach())

                if mu_anchor is not None:
                    anchor_per = (mu_s_shared[sel] - mu_anchor[sel]).pow(2).mean(dim=(1, 2))
                    step_anchor_sum = step_anchor_sum + ((1.0 - c) * anchor_per).sum()

                with torch.no_grad():
                    gap = self._waypoint_gap_m(student_planner, x0_s[sel], x0_t)
                    step_gap_sum = step_gap_sum + gap * len(indices)
                    if step == num_steps - 1:
                        last_teacher_x0_full[sel] = x0_t.float()
                n_used += len(indices)

            step_kd = step_kd_sum / max(n_used, 1)
            step_anchor = step_anchor_sum / max(n_used, 1)
            total_kd = total_kd + step_kd
            total_anchor = total_anchor + step_anchor
            kl_steps.append(step_kd.detach())
            x0_gap_steps.append((step_gap_sum / max(n_used, 1)).detach())
            last_student_x0 = x0_s

        kd_loss = total_kd / max(num_steps, 1)
        anchor_loss = total_anchor / max(num_steps, 1)
        pred_traj_s = student_planner.denorm_odo(last_student_x0.float())
        smooth_loss = self._jerk_loss(pred_traj_s)

        loss = (
            self.goal_loss_weight * goal_loss
            + self.kd_weight * kd_loss
            + self.anchor_weight * anchor_loss
            + self.smooth_weight * smooth_loss
        )

        if hasattr(student_planner, "eta") and hasattr(student_planner.eta, "eta_logit"):
            eta_logit = student_planner.eta.eta_logit
            loss = loss + torch.nan_to_num(
                eta_logit, nan=0.0, posinf=0.0, neginf=0.0
            ).sum() * 0.0
        if not torch.isfinite(loss):
            loss = self._finite_anchor_loss(student_planner)

        with torch.no_grad():
            traj_err = (pred_traj_s[..., :2] - gt_traj[..., :2]).norm(dim=-1)
            student_ade = traj_err.mean()
            student_fde = traj_err[:, -1].mean()
            goal_fde = goal_err_xy.mean()

        data = {
            "loss": loss,
            "distill_loss": kd_loss.detach(),
            "reverse_kl_loss": kd_loss.detach(),
            "goal_loss": goal_loss.detach(),
            "weighted_goal_loss": (self.goal_loss_weight * goal_loss).detach(),
            "anchor_loss": anchor_loss.detach(),
            "weighted_anchor_loss": (self.anchor_weight * anchor_loss).detach(),
            "smooth_loss": smooth_loss.detach(),
            "weighted_smooth_loss": (self.smooth_weight * smooth_loss).detach(),
            "goal_fde_m": goal_fde.detach(),
            "recoverability_mean": recoverability.mean().detach(),
            "recoverability_min": recoverability.min().detach(),
            "recoverability_max": recoverability.max().detach(),
            "student_fde_gt_m": student_fde.detach(),
            "student_ade_gt_m": student_ade.detach(),
            "x0_gap_m": torch.stack(x0_gap_steps).mean(),
            "gauss_kl_mean": torch.stack(kl_steps).mean(),
            "entropy_mean": torch.stack(entropy_steps).mean(),
            "sigma_mean": torch.stack(sigma_list).mean().detach(),
            "chain_abs_max": chain.float().abs().max().detach(),
            "denoising_steps": torch.tensor(float(num_steps), device=device),
            "pred_goal_x_mean": pred_goal[:, 0].float().mean().detach(),
            "pred_goal_y_mean": pred_goal[:, 1].float().mean().detach(),
        }
        zero = torch.tensor(0.0, device=device)
        for bucket in self.bucket_names:
            vals = per_bucket_kl[bucket]
            data[f"kl_{bucket}_mean"] = torch.stack(vals).mean() if vals else zero
            data[f"n_samples_{bucket}"] = torch.tensor(
                float(len(bucket_to_indices.get(bucket, []))), device=device
            )

        if self.collect_viz:
            self.last_viz = {
                "gt_traj": gt_traj.detach().cpu(),
                "goal": gt_goal.detach().cpu(),
                "pred_goal": pred_goal.detach().float().cpu(),
                "student_traj": pred_traj_s.detach().float().cpu(),
                "teacher_traj": student_planner.denorm_odo(last_teacher_x0_full).detach().float().cpu(),
                "buckets": list(resolved),
            }

        return BatchFeature(data=data)

    @staticmethod
    def _waypoint_gap_m(planner, x0_a: torch.Tensor, x0_b: torch.Tensor) -> torch.Tensor:
        a = planner.denorm_odo(x0_a.float())
        b = planner.denorm_odo(x0_b.float())
        return (a[..., :2] - b[..., :2]).norm(dim=-1).mean()

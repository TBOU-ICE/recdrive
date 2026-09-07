"""GoalBridge OPD with only goal recovery and ungated reverse-KL.

This is intentionally separate from ``recogdrive_goalbridge_distill_trainer``:
the original recoverability-gated KD, inverse-recoverability anchor and jerk
regularizer remain unchanged for existing experiments.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_goalbridge_distill_trainer import (
    ReCogDriveGoalBridgeDistillTrainer,
)


class ReCogDriveGoalBridgeGoalKDDistillTrainer(ReCogDriveGoalBridgeDistillTrainer):
    """Goal regression plus ungated, scene-routed reverse-transition KL."""

    def __init__(
        self,
        *args,
        goal_loss_weight: float = 1.0,
        kd_weight: float = 1.0,
        **kwargs,
    ):
        # Keep inherited DDIM helpers and numerical safeguards, but disable the
        # losses owned by the original trainer. compute_loss below does not
        # calculate or add anchor/smooth losses.
        super().__init__(
            *args,
            goal_loss_weight=goal_loss_weight,
            kd_weight=kd_weight,
            anchor_weight=0.0,
            **kwargs,
        )

    def compute_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        ref_planner: Optional[torch.nn.Module],
        vl_features: torch.Tensor,
        action_input,
        bucket_per_sample: List[str],
        anchor_planner: Optional[torch.nn.Module] = None,
    ) -> BatchFeature:
        del ref_planner, anchor_planner
        if not hasattr(student_planner, "predict_goal_from_encoded"):
            raise TypeError("GoalBridge requires PredictedGoalDiffusionPlanner as the student.")

        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature
        gt_traj = action_input.action.float()
        gt_goal = gt_traj[:, -1, :].contiguous()

        # Student predicts its deployable goal from observable inputs.
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, student_dtype
        )
        pred_goal, pred_goal_norm = student_planner.predict_goal_from_encoded(
            vl_s, his_s, ego_s
        )
        gt_goal_norm = student_planner.norm_odo(gt_goal.unsqueeze(1)).squeeze(1)
        gt_goal_norm = gt_goal_norm.to(pred_goal_norm.dtype)
        use_heading = bool(
            getattr(getattr(student_planner, "goal_encoder", None), "use_heading", False)
        )
        goal_dims = 3 if use_heading else 2
        goal_loss = F.smooth_l1_loss(
            pred_goal_norm[..., :goal_dims],
            gt_goal_norm[..., :goal_dims],
            reduction="mean",
        )

        # Collect states from the current student policy without differentiating
        # through the rollout. KD is evaluated with gradients at these states.
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

        resolved = [self._resolve_bucket(bucket) for bucket in bucket_per_sample]
        bucket_to_indices: Dict[str, List[int]] = {}
        for i, bucket in enumerate(resolved):
            bucket_to_indices.setdefault(bucket, []).append(i)

        total_kd = vl_features.new_zeros((), dtype=torch.float32)
        kd_sample_max = vl_features.new_zeros((), dtype=torch.float32)
        sigma_list: List[torch.Tensor] = []
        kl_steps: List[torch.Tensor] = []
        x0_gap_steps: List[torch.Tensor] = []
        per_bucket_kl: Dict[str, List[torch.Tensor]] = {
            bucket: [] for bucket in self.bucket_names
        }
        last_student_x0 = None
        last_teacher_x0_full = torch.zeros(
            batch_size, chain.shape[2], chain.shape[3], device=device
        )
        diagnostic_steps = {0, num_steps // 2, num_steps - 1}
        teacher_step_metrics = {}
        viz_step_predictions = []

        for step in range(num_steps):
            z_t = chain[:, step].to(student_dtype)
            t_batch = student_planner.make_timesteps(
                batch_size, ddim_t_list[step], device
            )
            idx_batch = student_planner.make_timesteps(batch_size, step, device)

            with student_planner.goal_context(pred_goal):
                _, logvar_s, x0_s = student_planner.p_mean_variance(
                    z_t,
                    t_batch,
                    idx_batch,
                    vl_s,
                    his_s,
                    ego_s,
                    deterministic=False,
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
            sigma_list.append(sigma_geom.float().mean().detach())

            step_kd_sum = vl_features.new_zeros((), dtype=torch.float32)
            step_gap_sum = vl_features.new_zeros((), dtype=torch.float32)
            n_used = 0
            teacher_x0_full = torch.zeros_like(last_teacher_x0_full)

            for bucket, indices in bucket_to_indices.items():
                sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                teacher = teacher_planners[bucket]
                enc = teacher_encodings[bucket]
                with torch.no_grad():
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

                # No recoverability gate: every sample receives full teacher KD,
                # including samples whose predicted goal is currently inaccurate.
                kl_per = (
                    (mu_s_shared[sel] - mu_t_shared).pow(2) * precision[sel]
                ).mean(dim=(1, 2))
                step_kd_sum = step_kd_sum + kl_per.sum()
                kd_sample_max = torch.maximum(
                    kd_sample_max, kl_per.detach().max()
                )
                per_bucket_kl[bucket].append(kl_per.mean().detach())

                with torch.no_grad():
                    gap = self._waypoint_gap_m(student_planner, x0_s[sel], x0_t)
                    step_gap_sum = step_gap_sum + gap * len(indices)
                    teacher_x0_full[sel] = x0_t.float()
                n_used += len(indices)

            step_kd = step_kd_sum / max(n_used, 1)
            total_kd = total_kd + step_kd
            kl_steps.append(step_kd.detach())
            x0_gap_steps.append((step_gap_sum / max(n_used, 1)).detach())
            last_student_x0 = x0_s
            if step in diagnostic_steps:
                with torch.no_grad():
                    student_step_traj = student_planner.denorm_odo(x0_s.float())
                    teacher_step_traj = student_planner.denorm_odo(
                        teacher_x0_full
                    )
                    teacher_step_err = (
                        teacher_step_traj[..., :2] - gt_traj[..., :2]
                    ).norm(dim=-1)
                    teacher_step_metrics[
                        f"fde_gt_teacher_step_{step}_m"
                    ] = teacher_step_err[:, -1].mean().detach()
                    teacher_step_metrics[
                        f"fde_gt_teacher_step_{step}_ade_m"
                    ] = teacher_step_err.mean().detach()
                    if self.collect_viz:
                        viz_step_predictions.append(
                            {
                                "step": step,
                                "student_x0": student_step_traj.detach().cpu(),
                                "teacher_x0": teacher_step_traj.detach().cpu(),
                            }
                        )
            if step == num_steps - 1:
                last_teacher_x0_full = teacher_x0_full

        kd_loss = total_kd / max(num_steps, 1)
        loss = self.goal_loss_weight * goal_loss + self.kd_weight * kd_loss

        # eta is detached from the precision and may otherwise be unused in DDP.
        if hasattr(student_planner, "eta") and hasattr(
            student_planner.eta, "eta_logit"
        ):
            eta_logit = student_planner.eta.eta_logit
            loss = loss + torch.nan_to_num(
                eta_logit, nan=0.0, posinf=0.0, neginf=0.0
            ).sum() * 0.0
        if not torch.isfinite(loss):
            loss = self._finite_anchor_loss(student_planner)

        pred_traj_s = student_planner.denorm_odo(last_student_x0.float())
        with torch.no_grad():
            teacher_traj_final = student_planner.denorm_odo(
                last_teacher_x0_full
            )
            traj_err = (
                pred_traj_s[..., :2] - gt_traj[..., :2]
            ).norm(dim=-1)
            teacher_err = (
                teacher_traj_final[..., :2] - gt_traj[..., :2]
            ).norm(dim=-1)
            goal_fde = (
                pred_goal.float()[..., :2] - gt_goal[..., :2]
            ).norm(dim=-1).mean()

        data = {
            "loss": loss,
            "distill_loss": kd_loss.detach(),
            "reverse_kl_loss": kd_loss.detach(),
            "goal_loss": goal_loss.detach(),
            "weighted_goal_loss": (
                self.goal_loss_weight * goal_loss
            ).detach(),
            "weighted_kd_loss": (self.kd_weight * kd_loss).detach(),
            "goal_fde_m": goal_fde.detach(),
            "student_fde_gt_m": traj_err[:, -1].mean().detach(),
            "student_ade_gt_m": traj_err.mean().detach(),
            # Prefix with fde_gt_ so the existing goal Lightning wrapper logs
            # these optional diagnostics without changing its fixed key list.
            "fde_gt_teacher_final_m": teacher_err[:, -1].mean().detach(),
            "fde_gt_teacher_ade_m": teacher_err.mean().detach(),
            "x0_gap_m": torch.stack(x0_gap_steps).mean(),
            "gauss_kl_mean": torch.stack(kl_steps).mean(),
            "kl_ungated_sample_max": kd_sample_max,
            "sigma_mean": torch.stack(sigma_list).mean().detach(),
            "chain_abs_max": chain.float().abs().max().detach(),
            "denoising_steps": torch.tensor(float(num_steps), device=device),
            "pred_goal_x_mean": pred_goal[:, 0].float().mean().detach(),
            "pred_goal_y_mean": pred_goal[:, 1].float().mean().detach(),
        }
        data.update(teacher_step_metrics)
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
                "pred_goal": pred_goal.detach().float().cpu(),
                "student_traj": pred_traj_s.detach().float().cpu(),
                "teacher_traj": teacher_traj_final.detach().float().cpu(),
                "step_predictions": viz_step_predictions,
                "buckets": list(resolved),
            }

        return BatchFeature(data=data)

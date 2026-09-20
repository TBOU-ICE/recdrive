"""Isolated KD-only GoalBridge OPD trainer with per-step diagnostics.

This file deliberately leaves the existing gated GoalBridge and goal+KD
trainers unchanged.  The student rolls out on policy with its predicted goal;
the routed privileged teacher is evaluated at the same student state with the
GT endpoint.  All samples receive full-strength reverse-KL mean matching.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_distill_trainer import (
    ReCogDriveGoalBridgeGoalKDDistillTrainer,
)


class ReCogDriveGoalBridgeKDOnlyDistillTrainer(
    ReCogDriveGoalBridgeGoalKDDistillTrainer
):
    """Ungated reverse-KL only, plus all-step KL/ADE/FDE diagnostics."""

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
            raise TypeError(
                "GoalBridge KD-only requires PredictedGoalDiffusionPlanner."
            )

        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature
        gt_traj = action_input.action.float()
        gt_goal = gt_traj[:, -1, :].contiguous()

        # The goal loss is retained only as an unweighted diagnostic.  The
        # optimization objective below contains KD and nothing else.
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, student_dtype
        )
        pred_goal, pred_goal_norm = student_planner.predict_goal_from_encoded(
            vl_s, his_s, ego_s
        )
        gt_goal_norm = student_planner.norm_odo(gt_goal.unsqueeze(1)).squeeze(1)
        gt_goal_norm = gt_goal_norm.to(pred_goal_norm.dtype)
        use_heading = bool(
            getattr(
                getattr(student_planner, "goal_encoder", None),
                "use_heading",
                False,
            )
        )
        goal_dims = 3 if use_heading else 2
        goal_loss = F.smooth_l1_loss(
            pred_goal_norm[..., :goal_dims],
            gt_goal_norm[..., :goal_dims],
            reduction="mean",
        )

        # Reverse-KL expectation: sample states from the current student.  Do
        # not backpropagate through the rollout; recompute the student x0 with
        # gradients at every visited state below.
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
        # Match the reference visualisation: early, middle and final x0 pairs.
        viz_steps = {0, num_steps // 2, num_steps - 1}
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

            # Keep the established OPD surrogate: both reverse means use the
            # student's stochastic DDIM sigma.  The floor affects only the KL
            # precision, never the reverse-process geometry.
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
            total_kd = total_kd + step_kd
            kl_steps.append(step_kd.detach())
            x0_gap_steps.append(step_gap.detach())
            sigma_steps.append(sigma_geom.float().mean().detach())
            last_student_x0 = x0_s

            # All five steps get KD, student/teacher ADE and FDE diagnostics.
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
        # Strict KD-only objective: goal_loss is diagnostic and has no gradient
        # contribution.  KD still reaches the goal predictor through pred_goal.
        loss = self.kd_weight * kd_loss

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
            student_err = (
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
            "weighted_goal_loss": goal_loss.detach() * 0.0,
            "goal_fde_m": goal_fde.detach(),
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
            "pred_goal_x_mean": pred_goal[:, 0].float().mean().detach(),
            "pred_goal_y_mean": pred_goal[:, 1].float().mean().detach(),
        }
        data.update(step_metrics)

        # These names start with kl_, so the existing Lightning wrapper records
        # them without any shared-code change.  Fractions directly reveal which
        # denoising step dominates the five-step average.
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
                "pred_goal": pred_goal.detach().float().cpu(),
                "student_traj": pred_traj_s.detach().float().cpu(),
                "teacher_traj": teacher_traj_final.detach().float().cpu(),
                "step_predictions": viz_step_predictions,
                "buckets": list(resolved),
            }

        return BatchFeature(data=data)

"""
Scene-router OPD distillation with GOAL-CONDITIONED (privileged) teachers.

Additive file: does not modify any existing implementation.  Subclasses
``ReCogDriveDiTSceneRouterDistillTrainer`` and changes exactly one thing about
the loss path: every teacher ``p_mean_variance`` call runs inside
``teacher.goal_context(goal[sel])``, where ``goal`` is the ground-truth
trajectory endpoint (``action_input.action[:, -1, :]`` -- the same quantity the
goal teachers were trained with).  The student and the optional ExOPD ref stay
goal-free, so the privileged information flows exclusively through the
distillation target: the student must learn to reproduce goal-informed
behaviour from scene features alone.

On top of that it logs distillation diagnostics (all scalars ride the existing
lightning logging; per-bucket keys are emitted for EVERY bucket every step so
DDP collectives stay rank-symmetric):

teacher/student agreement
    ``x0_gap_m`` / ``x0_gap_final_m``: mean waypoint distance (metres, denormed)
    between student and routed-teacher x0, chain-averaged / at the final step.
    ``gauss_kl_mean``: shared-variance Gaussian KL between the student and
    teacher transition means, sum over the trajectory, using the student's
    sigma (the teacher runs deterministic, so its own sigma is degenerate).
    This is the driving analogue of OPD's per-token reverse KL.

student rollout quality
    ``entropy_mean`` / ``entropy_step_{i}``: student transition entropy
    H = sum 0.5 * log(2*pi*e*sigma^2) per chain step -- early collapse or
    non-decreasing entropy are both visible at a glance.
    ``student_fde_gt_m`` / ``fde_gt_{bucket}``: distance (m) from the student's
    final-trajectory endpoint to the GT goal.  The student never sees the goal,
    so this is the most direct online measure of privileged-information
    internalisation.
    ``student_ade_gt_m``: ADE (m) of the student's final trajectory vs GT.

goal liveness
    ``teacher_goal_effect_m`` / ``goal_effect_{bucket}``: on the final chain
    step, mean waypoint distance (m) between the teacher's x0 with and without
    the goal bound.  Near-zero means that teacher's goal branch is dead
    (wrong goal_mode / silently dropped weights) -- the failure mode that cost
    us the 2026.08.02 channel/cross teachers.  One extra no-grad forward per
    bucket per step, only at the final chain step.

For BEV visualisation the trainer stashes the last batch's trajectories in
``self.last_viz`` (small CPU tensors); the goal lightning module decides when
to draw them.
"""

from typing import Dict, List, Optional

import math

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_dit_scene_router_distill_trainer import (
    ReCogDriveDiTSceneRouterDistillTrainer,
)

_LOG_2PI_E = math.log(2.0 * math.pi * math.e)


class ReCogDriveDiTSceneRouterGoalDistillTrainer(ReCogDriveDiTSceneRouterDistillTrainer):
    """Scene-routed OPD whose teachers are conditioned on the privileged GT goal."""

    def __init__(self, *args, collect_viz: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_viz = bool(collect_viz)
        # Written by compute_loss, read (and cleared) by the lightning module.
        self.last_viz: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------- loss
    def compute_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        ref_planner: Optional[torch.nn.Module],
        vl_features: torch.Tensor,
        action_input,
        bucket_per_sample: List[str],
    ) -> BatchFeature:
        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature

        # Privileged goal = GT trajectory endpoint, exactly what the goal
        # teachers consumed during their own training (raw ego-frame metres;
        # each teacher normalises internally in encode_goal).
        gt_traj = action_input.action.float()          # (B, H, 3), raw metres
        goal = gt_traj[:, -1, :].contiguous()          # (B, 3)

        use_exopd = ref_planner is not None and abs(self.exopd_lambda - 1.0) > 1e-6
        use_x0 = self.match_target == "x0"

        # 1) on-policy detached student DDIM chain (goal-free, by construction)
        with torch.no_grad():
            vl_s0, his_s0, ego_s0 = self._encode(
                student_planner, vl_features, his_traj, ego_status, student_dtype
            )
            chain = self._sample_chain(
                student_planner, vl_s0, his_s0, ego_s0, batch_size, device, student_dtype
            )
        num_steps = chain.shape[1] - 1
        ddim_t_list = [int(student_planner.ddim_t[i].item()) for i in range(num_steps)]

        # 2) teacher / ref encodings (frozen, float32)
        teacher_encodings: Dict[str, tuple] = {}
        with torch.no_grad():
            for name, teacher in teacher_planners.items():
                teacher.eval()
                teacher_encodings[name] = self._encode(
                    teacher, vl_features, his_traj, ego_status, torch.float32
                )
            ref_encoding = (
                self._encode(ref_planner, vl_features, his_traj, ego_status, torch.float32)
                if use_exopd else None
            )

        # 3) student encodings (with grad)
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, student_dtype
        )

        resolved = [self._resolve_bucket(b) for b in bucket_per_sample]
        bucket_to_indices: Dict[str, List[int]] = {}
        for i, b in enumerate(resolved):
            bucket_to_indices.setdefault(b, []).append(i)

        total_loss = vl_features.new_zeros(())
        per_bucket_step_losses: Dict[str, list] = {b: [] for b in self.bucket_names}
        sigma_list = []
        last_student_x0 = None

        # metric accumulators
        entropy_steps: List[torch.Tensor] = []
        x0_gap_steps: List[torch.Tensor] = []
        gauss_kl_steps: List[torch.Tensor] = []
        x0_gap_final = vl_features.new_zeros(())
        goal_effect_per_bucket: Dict[str, torch.Tensor] = {}
        last_teacher_x0_full = torch.zeros(
            batch_size, chain.shape[2], chain.shape[3], device=device
        )

        for step in range(num_steps):
            is_last = step == num_steps - 1
            z_t = chain[:, step].to(student_dtype)
            t_batch = student_planner.make_timesteps(batch_size, ddim_t_list[step], device)
            idx_batch = student_planner.make_timesteps(batch_size, step, device)

            mu_s, logvar_s, x0_s = student_planner.p_mean_variance(
                z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False
            )
            sigma_s = self._safe_sigma(logvar_s, self.min_sigma, student_dtype).detach()
            sigma2 = sigma_s.float().pow(2).clamp(min=1e-6)
            sigma_list.append(sigma_s.detach().float().mean())
            student_side = (x0_s if use_x0 else mu_s).float()

            # student transition entropy over the full (H x D)-dim Gaussian:
            # H = sum_i 0.5 * log(2*pi*e*sigma_i^2). The DDIM sigma comes back
            # broadcast as (B, 1, 1), so mean over the sigma dims and scale by
            # the true dimensionality (correct for full-shape sigmas too).
            traj_dims = float(x0_s.shape[1] * x0_s.shape[2])
            entropy_steps.append(
                ((0.5 * (_LOG_2PI_E + sigma2.log())).mean(dim=(1, 2)) * traj_dims)
                .mean()
                .detach()
            )

            ref_side_full = None
            if use_exopd:
                with torch.no_grad():
                    mu_r, _, x0_r = ref_planner.p_mean_variance(
                        z_t.float(), t_batch, idx_batch,
                        ref_encoding[0], ref_encoding[1], ref_encoding[2],
                        deterministic=True,
                    )
                    ref_side_full = (x0_r if use_x0 else mu_r).detach()

            step_loss = vl_features.new_zeros(())
            n_used = 0
            step_gap_sum = vl_features.new_zeros(())
            step_kl_sum = vl_features.new_zeros(())
            for bucket, indices in bucket_to_indices.items():
                enc = teacher_encodings[bucket]
                teacher = teacher_planners[bucket]
                sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                z_sel = z_t[sel].float()
                t_sel = t_batch[sel]
                idx_sel = idx_batch[sel]
                goal_sel = goal[sel]
                with torch.no_grad():
                    # >>> the one functional change vs the goal-free trainer:
                    # the teacher denoises the student's state CONDITIONED ON
                    # the privileged GT goal it was trained with.
                    with teacher.goal_context(goal_sel):
                        mu_t, _, x0_t = teacher.p_mean_variance(
                            z_sel, t_sel, idx_sel,
                            enc[0][sel], enc[1][sel], enc[2][sel],
                            deterministic=True,
                        )
                    teacher_side = (x0_t if use_x0 else mu_t).detach()
                    if use_exopd:
                        ref_side = ref_side_full[sel]
                        target = ref_side + self.exopd_lambda * (teacher_side - ref_side)
                    else:
                        target = teacher_side

                    # -------- diagnostics (no_grad) --------
                    gap_m = self._waypoint_gap_m(student_planner, x0_s[sel], x0_t)
                    step_gap_sum = step_gap_sum + gap_m * len(indices)
                    kl = (
                        (mu_s[sel].float() - mu_t.float()).pow(2)
                        .div(2.0 * sigma2[sel])
                        .sum(dim=(1, 2))
                        .mean()
                    )
                    step_kl_sum = step_kl_sum + kl * len(indices)

                    if is_last:
                        last_teacher_x0_full[sel] = x0_t.float()
                        # goal liveness: same z_t, goal unbound
                        _, _, x0_t_off = teacher.p_mean_variance(
                            z_sel, t_sel, idx_sel,
                            enc[0][sel], enc[1][sel], enc[2][sel],
                            deterministic=True,
                        )
                        goal_effect_per_bucket[bucket] = self._waypoint_gap_m(
                            student_planner, x0_t, x0_t_off
                        ).detach()

                diff = student_side[sel] - target
                loss_per_sample = diff.pow(2).div(2.0 * sigma2[sel]).sum(dim=(1, 2))
                per_bucket_step_losses[bucket].append(loss_per_sample.mean().detach())
                step_loss = step_loss + loss_per_sample.sum()
                n_used += len(indices)

            total_loss = total_loss + step_loss / max(n_used, 1)
            x0_gap_steps.append((step_gap_sum / max(n_used, 1)).detach())
            gauss_kl_steps.append((step_kl_sum / max(n_used, 1)).detach())
            if is_last:
                last_student_x0 = x0_s
                x0_gap_final = x0_gap_steps[-1]

        distill_loss = total_loss / num_steps
        pred_traj_s = student_planner.denorm_odo(last_student_x0.float())
        smooth_loss = self._jerk_loss(pred_traj_s)
        loss = distill_loss + self.smooth_weight * smooth_loss
        # See base trainer: keep the eta parameter inside the DDP graph with a
        # zero coefficient. eta_logit is initialised to atanh(1.0)=+inf whenever
        # base_eta==max_eta (EtaFixed), and that inf is carried in the student
        # checkpoint, so a plain ``eta_logit.sum() * 0.0`` would be ``inf * 0 =
        # NaN`` and silently poison every step's loss into the zero fallback
        # below (observed: the student never trained for a full run). nan_to_num
        # keeps the graph edge while guaranteeing a finite zero contribution.
        if hasattr(student_planner, "eta") and hasattr(student_planner.eta, "eta_logit"):
            eta_logit = student_planner.eta.eta_logit
            loss = loss + torch.nan_to_num(eta_logit, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        if not torch.isfinite(loss):
            # last_student_x0 is already NaN in the crash we observed; NaN*0
            # stays NaN and poisons AdamW under bf16-mixed. Anchor on params.
            loss = self._finite_anchor_loss(student_planner)

        # -------- student-vs-GT quality (student never saw the goal) --------
        with torch.no_grad():
            err_xy = (pred_traj_s[..., :2] - gt_traj[..., :2]).norm(dim=-1)  # (B, H)
            student_ade = err_xy.mean()
            student_fde = err_xy[:, -1].mean()
            fde_per_bucket = {
                bucket: err_xy[torch.as_tensor(idxs, device=device), -1].mean().detach()
                for bucket, idxs in bucket_to_indices.items()
            }

        data = {
            "loss": loss,
            "distill_loss": distill_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
            "weighted_smooth_loss": (self.smooth_weight * smooth_loss).detach(),
            "sigma_mean": torch.stack(sigma_list).mean().detach(),
            "chain_abs_max": chain.float().abs().max().detach(),
            "denoising_steps": torch.tensor(float(num_steps), device=device),
            "exopd_lambda": torch.tensor(float(self.exopd_lambda), device=device),
            "student_pred_traj_mean": pred_traj_s.mean().detach(),
            "student_pred_traj_std": pred_traj_s.std(unbiased=False).detach(),
            # distillation diagnostics (batch-independent key set on every rank)
            "x0_gap_m": torch.stack(x0_gap_steps).mean(),
            "x0_gap_final_m": x0_gap_final,
            "gauss_kl_mean": torch.stack(gauss_kl_steps).mean(),
            "entropy_mean": torch.stack(entropy_steps).mean(),
            "student_fde_gt_m": student_fde.detach(),
            "student_ade_gt_m": student_ade.detach(),
            "teacher_goal_effect_m": (
                torch.stack(list(goal_effect_per_bucket.values())).mean()
                if goal_effect_per_bucket
                else torch.tensor(0.0, device=device)
            ),
        }
        for i, ent in enumerate(entropy_steps):
            data[f"entropy_step_{i}"] = ent

        # per-bucket keys: ALWAYS emit every bucket (0 when absent) so the key
        # set is identical across DDP ranks (see the kl_ note in the base file).
        zero = torch.tensor(0.0, device=device)
        for bucket in self.bucket_names:
            values = per_bucket_step_losses[bucket]
            data[f"kl_{bucket}_mean"] = torch.stack(values).mean().detach() if values else zero
            data[f"n_samples_{bucket}"] = torch.tensor(
                float(len(bucket_to_indices.get(bucket, []))), device=device
            )
            data[f"fde_gt_{bucket}"] = fde_per_bucket.get(bucket, zero)
            data[f"goal_effect_{bucket}"] = goal_effect_per_bucket.get(bucket, zero)

        if self.collect_viz:
            with torch.no_grad():
                self.last_viz = {
                    "gt_traj": gt_traj.detach().cpu(),
                    "goal": goal.detach().cpu(),
                    "student_traj": pred_traj_s.detach().float().cpu(),
                    "teacher_traj": student_planner.denorm_odo(last_teacher_x0_full)
                    .detach().float().cpu(),
                    "buckets": list(resolved),
                }

        return BatchFeature(data=data)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _waypoint_gap_m(planner, x0_a: torch.Tensor, x0_b: torch.Tensor) -> torch.Tensor:
        """Mean per-waypoint xy distance (metres) between two normalised x0."""
        a = planner.denorm_odo(x0_a.float())
        b = planner.denorm_odo(x0_b.float())
        return (a[..., :2] - b[..., :2]).norm(dim=-1).mean()

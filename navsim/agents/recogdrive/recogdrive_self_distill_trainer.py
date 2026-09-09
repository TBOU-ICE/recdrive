"""Privileged self-distillation for the goal-conditioned DiT planner.

One set of weights plays both roles.  Every batch runs the *same* planner twice
on the *same* on-policy DDIM states:

    teacher pass : conditioned on the ground-truth endpoint (privileged)
    student pass : conditioned on the endpoint its own goal head predicted

so the classic OPD "model mismatch" term of the realizability gap is identically
zero and only the privileged-information term is left.  That term is exactly the
skill we want: infer the intent that the GT endpoint reveals, from observations
alone.

The objective has three terms and each one is load bearing:

``il``    behaviour cloning on the teacher pass.  Without it the trivial optimum
          is to ignore the goal channel entirely, which makes the KD term zero
          and the teacher useless.  This is the anti-collapse anchor and it
          replaces the frozen goal-free anchor planner of GoalBridge OPD.
``kd``    shared-variance reverse-transition KL from the student pass to the
          teacher pass, evaluated at states the student actually visits.
``goal``  regression of the predicted endpoint onto the GT endpoint; the only
          direct signal that teaches the missing privileged information.

Unlike GoalBridge OPD there is no recoverability gate: with a self-teacher a
badly predicted goal no longer drags the student toward a *foreign* policy, it
drags it toward its own behaviour under the correct intent, which is exactly
what should happen.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_goalbridge_distill_trainer import (
    ReCogDriveGoalBridgeDistillTrainer,
)


class ReCogDriveSelfDistillTrainer(ReCogDriveGoalBridgeDistillTrainer):
    """Single-model privileged self-distillation (IL + reverse-KL + goal)."""

    def __init__(
        self,
        *args,
        il_weight: float = 1.0,
        kd_weight: float = 1.0,
        goal_loss_weight: float = 1.0,
        teacher_goal_dropout_p: float = 0.0,
        teacher_goal_noise_p: float = 0.0,
        teacher_goal_noise_std_xy: float = 1.0,
        kd_warmup_steps: int = 0,
        goal_probe_interval: int = 50,
        goal_probe_shift_m: float = 2.0,
        **kwargs,
    ):
        kwargs.pop("anchor_weight", None)
        super().__init__(
            *args,
            goal_loss_weight=goal_loss_weight,
            kd_weight=kd_weight,
            anchor_weight=0.0,
            **kwargs,
        )
        self.il_weight = float(il_weight)
        self.teacher_goal_dropout_p = float(teacher_goal_dropout_p)
        self.teacher_goal_noise_p = float(teacher_goal_noise_p)
        self.teacher_goal_noise_std_xy = float(teacher_goal_noise_std_xy)
        self.kd_warmup_steps = int(kd_warmup_steps)
        self.goal_probe_interval = int(goal_probe_interval)
        self.goal_probe_shift_m = float(goal_probe_shift_m)
        self._step_counter = 0

    # --------------------------------------------------------------- helpers
    @contextmanager
    def _teacher_goal_corruption(self, planner):
        """Enable the planner's built-in goal mask/noise for the teacher pass only.

        A teacher that only ever sees a perfect endpoint becomes an oracle the
        student can never catch: that is exactly how the separately trained
        privileged teacher ended up scoring 70 EPDMS on the student's predicted
        goal.  Corrupting the goal at the noise level the goal head actually has
        keeps the teacher inside the student's reachable set.  The student pass
        and the on-policy rollout must stay uncorrupted, hence the scoping.
        """
        if self.teacher_goal_dropout_p <= 0.0 and self.teacher_goal_noise_p <= 0.0:
            yield planner
            return
        saved = (planner.goal_dropout_p, planner.goal_noise_p, planner.goal_noise_std_xy)
        planner.goal_dropout_p = self.teacher_goal_dropout_p
        planner.goal_noise_p = self.teacher_goal_noise_p
        planner.goal_noise_std_xy = self.teacher_goal_noise_std_xy
        try:
            yield planner
        finally:
            (
                planner.goal_dropout_p,
                planner.goal_noise_p,
                planner.goal_noise_std_xy,
            ) = saved

    # ------------------------------------------------------------------ loss
    def compute_loss(
        self,
        student_planner,
        teacher_planners: Optional[Dict[str, torch.nn.Module]] = None,
        ref_planner: Optional[torch.nn.Module] = None,
        vl_features: torch.Tensor = None,
        action_input=None,
        bucket_per_sample: Optional[List[str]] = None,
        anchor_planner: Optional[torch.nn.Module] = None,
    ) -> BatchFeature:
        # Signature kept compatible with the OPD trainers so the same Lightning
        # wrapper and agent plumbing work unchanged; no external model is used.
        del teacher_planners, ref_planner, anchor_planner
        if not hasattr(student_planner, "predict_goal_from_encoded"):
            raise TypeError("Self-distillation requires PredictedGoalDiffusionPlanner.")

        self._step_counter += 1
        planner = student_planner
        batch_size = vl_features.shape[0]
        device = vl_features.device
        dtype = next(planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature
        gt_traj = action_input.action.float()
        gt_goal = gt_traj[:, -1, :].contiguous()

        # One shared encoding: teacher and student differ only in the goal that
        # is injected into the DiT conditioning, never in the scene features.
        vl_e, his_e, ego_e = self._encode(planner, vl_features, his_traj, ego_status, dtype)

        pred_goal, pred_goal_norm = planner.predict_goal_from_encoded(vl_e, his_e, ego_e)
        gt_goal_norm = planner.norm_odo(gt_goal.unsqueeze(1)).squeeze(1).to(pred_goal_norm.dtype)
        use_heading = bool(getattr(getattr(planner, "goal_encoder", None), "use_heading", False))
        goal_dims = 3 if use_heading else 2
        goal_loss = F.smooth_l1_loss(
            pred_goal_norm[..., :goal_dims], gt_goal_norm[..., :goal_dims], reduction="mean"
        )

        # KD has two descent directions: improve pred_goal (wanted), or make the
        # policy goal-insensitive so both passes agree regardless (collapse).
        # The second is far cheaper -- the DiT trunk is large, the AdaLN goal
        # branch is thin -- and the IL term does NOT forbid it, because a
        # goal-blind but accurate policy satisfies IL on every unimodal scene.
        # Gating KD by how close the predicted goal already is removes exactly
        # that pressure: samples whose two conditionings are far apart, i.e. the
        # ones that would be "fixed" by deleting the goal channel, are muted.
        goal_err_xy = (pred_goal.float()[..., :2] - gt_goal[..., :2]).norm(dim=-1)
        if self.recoverability_tau_m > 0.0:
            tau = self.recoverability_tau_m
            recoverability = torch.exp(-(goal_err_xy.detach() ** 2) / (2.0 * tau * tau))
            recoverability = recoverability.clamp(min=self.recoverability_floor, max=1.0)
        else:
            recoverability = torch.ones_like(goal_err_xy)

        # On-policy states: sampled from the deployable (predicted-goal) policy,
        # detached so no gradient flows through the rollout.
        with torch.no_grad():
            with planner.goal_context(pred_goal.detach()):
                chain = self._sample_chain(
                    planner,
                    vl_e.detach(),
                    his_e.detach(),
                    ego_e.detach(),
                    batch_size,
                    device,
                    dtype,
                )
        num_steps = chain.shape[1] - 1
        ddim_t_list = [int(planner.ddim_t[i].item()) for i in range(num_steps)]

        total_kd = vl_features.new_zeros((), dtype=torch.float32)
        total_il = vl_features.new_zeros((), dtype=torch.float32)
        kl_steps: List[torch.Tensor] = []
        sigma_list: List[torch.Tensor] = []
        x0_gap_steps: List[torch.Tensor] = []
        last_student_x0 = None
        last_teacher_x0 = None
        # Early / middle / late states of the student's own chain. This is the
        # only view that answers "is the privileged pass actually better than the
        # deployable one, at the states the deployable one visits".
        diagnostic_steps = {0, num_steps // 2, num_steps - 1}
        teacher_step_metrics: Dict[str, torch.Tensor] = {}
        viz_step_predictions: List[Dict] = []

        for step in range(num_steps):
            z_t = chain[:, step].to(dtype)
            t_batch = planner.make_timesteps(batch_size, ddim_t_list[step], device)
            idx_batch = planner.make_timesteps(batch_size, step, device)

            with planner.goal_context(pred_goal):
                _, logvar_s, x0_s = planner.p_mean_variance(
                    z_t, t_batch, idx_batch, vl_e, his_e, ego_e, deterministic=False
                )
            # Same weights, same state, privileged goal. Gradients are kept here:
            # this pass carries the IL term that prevents goal-channel collapse.
            with self._teacher_goal_corruption(planner), planner.goal_context(gt_goal):
                _, _, x0_t = planner.p_mean_variance(
                    z_t, t_batch, idx_batch, vl_e, his_e, ego_e, deterministic=False
                )

            sigma_geom = (0.5 * logvar_s.float().clamp(-40.0, 20.0)).exp().detach()
            sigma_weight = sigma_geom.clamp(min=self.min_sigma)
            precision = self._precision(sigma_weight.pow(2).clamp(min=1e-6))
            sigma_list.append(sigma_geom.float().mean().detach())

            mu_s = self._shared_ddim_mean(planner, z_t, idx_batch, x0_s, sigma_geom)
            # The KD target must not be trainable, otherwise the cheapest descent
            # direction is to move the teacher toward the student.
            mu_t = self._shared_ddim_mean(planner, z_t, idx_batch, x0_t.detach(), sigma_geom)
            kd_per_sample = ((mu_s - mu_t).pow(2) * precision).mean(dim=(1, 2))
            total_kd = total_kd + (kd_per_sample * recoverability).mean()

            # IL in metric space so its scale is comparable to the FDE numbers we
            # actually track, instead of normalised-unit noise.
            teacher_traj = planner.denorm_odo(x0_t.float())
            total_il = total_il + F.smooth_l1_loss(
                teacher_traj[..., :2], gt_traj[..., :2], reduction="mean"
            )

            with torch.no_grad():
                x0_gap_steps.append(self._waypoint_gap_m(planner, x0_s, x0_t))
            kl_steps.append(kd_per_sample.mean().detach())
            last_student_x0 = x0_s
            last_teacher_x0 = x0_t

            if step in diagnostic_steps:
                with torch.no_grad():
                    student_step_traj = planner.denorm_odo(x0_s.float())
                    teacher_step_traj = planner.denorm_odo(x0_t.float())
                    teacher_step_err = (
                        teacher_step_traj[..., :2] - gt_traj[..., :2]
                    ).norm(dim=-1)
                    teacher_step_metrics[f"fde_gt_teacher_step_{step}_m"] = (
                        teacher_step_err[:, -1].mean().detach()
                    )
                    teacher_step_metrics[f"fde_gt_teacher_step_{step}_ade_m"] = (
                        teacher_step_err.mean().detach()
                    )
                    if self.collect_viz:
                        viz_step_predictions.append({
                            "step": step,
                            "student_x0": student_step_traj.detach().cpu(),
                            "teacher_x0": teacher_step_traj.detach().cpu(),
                        })

        kd_loss = total_kd / max(num_steps, 1)
        il_loss = total_il / max(num_steps, 1)
        # Ramp KD in. At step 0 the goal head is randomly initialised, so every
        # sample has a huge goal error and KD is pure collapse pressure; let the
        # goal regression pull pred_goal into range first.
        kd_scale = self.kd_weight
        if self.kd_warmup_steps > 0:
            kd_scale = kd_scale * min(1.0, self._step_counter / self.kd_warmup_steps)
        loss = (
            self.il_weight * il_loss
            + kd_scale * kd_loss
            + self.goal_loss_weight * goal_loss
        )

        if hasattr(planner, "eta") and hasattr(planner.eta, "eta_logit"):
            eta_logit = planner.eta.eta_logit
            loss = loss + torch.nan_to_num(eta_logit, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        if not torch.isfinite(loss):
            loss = self._finite_anchor_loss(planner)

        with torch.no_grad():
            student_traj = planner.denorm_odo(last_student_x0.float())
            teacher_traj = planner.denorm_odo(last_teacher_x0.float())
            student_err = (student_traj[..., :2] - gt_traj[..., :2]).norm(dim=-1)
            teacher_err = (teacher_traj[..., :2] - gt_traj[..., :2]).norm(dim=-1)
            goal_fde = (pred_goal.float()[..., :2] - gt_goal[..., :2]).norm(dim=-1).mean()

        data = {
            "loss": loss,
            "distill_loss": kd_loss.detach(),
            "reverse_kl_loss": kd_loss.detach(),
            "il_loss": il_loss.detach(),
            "goal_loss": goal_loss.detach(),
            "weighted_il_loss": (self.il_weight * il_loss).detach(),
            "weighted_kd_loss": (kd_scale * kd_loss).detach(),
            "recoverability_mean": recoverability.mean().detach(),
            "weighted_goal_loss": (self.goal_loss_weight * goal_loss).detach(),
            # The number to tune kd_weight on. KD lives in normalised precision
            # units and IL in metres, so out of the box KD is orders of magnitude
            # smaller and contributes nothing -- the exact failure the previous
            # GoalBridge OPD run had. Aim for roughly 0.2-0.5.
            "kd_il_ratio": (
                (kd_scale * kd_loss) / (self.il_weight * il_loss + 1e-8)
            ).detach(),
            "goal_fde_m": goal_fde.detach(),
            "student_fde_gt_m": student_err[:, -1].mean().detach(),
            "student_ade_gt_m": student_err.mean().detach(),
            "fde_gt_teacher_final_m": teacher_err[:, -1].mean().detach(),
            "fde_gt_teacher_ade_m": teacher_err.mean().detach(),
            "x0_gap_m": torch.stack(x0_gap_steps).mean(),
            "gauss_kl_mean": torch.stack(kl_steps).mean(),
            "sigma_mean": torch.stack(sigma_list).mean().detach(),
            "chain_abs_max": chain.float().abs().max().detach(),
            "denoising_steps": torch.tensor(float(num_steps), device=device),
            "pred_goal_x_mean": pred_goal[:, 0].float().mean().detach(),
            "pred_goal_y_mean": pred_goal[:, 1].float().mean().detach(),
        }
        data.update(teacher_step_metrics)
        data.update(
            self._goal_sensitivity_probe(
                planner, chain, vl_e, his_e, ego_e, gt_goal, num_steps, batch_size, device, dtype
            )
        )

        if self.collect_viz:
            self.last_viz = {
                "gt_traj": gt_traj.detach().cpu(),
                "goal": gt_goal.detach().cpu(),
                "pred_goal": pred_goal.detach().float().cpu(),
                "student_traj": student_traj.detach().float().cpu(),
                "teacher_traj": teacher_traj.detach().float().cpu(),
                "step_predictions": viz_step_predictions,
                "buckets": ["gt-goal vs pred-goal"] * batch_size,
            }

        return BatchFeature(data=data)

    # ------------------------------------------------------------- diagnostics
    @torch.no_grad()
    def _goal_sensitivity_probe(
        self, planner, chain, vl_e, his_e, ego_e, gt_goal, num_steps, batch_size, device, dtype
    ) -> Dict[str, torch.Tensor]:
        """How many metres of endpoint motion does one metre of goal motion buy?

        This is the single number that says whether self-distillation is doing
        anything.  A ratio near 0 means the goal channel is dead and the teacher
        pass is identical to the student pass (collapse).  A ratio near 1 means
        the policy tracks the commanded endpoint.  Probed on a middle DDIM step
        because that is where the trajectory shape is already decided.
        """
        if self.goal_probe_interval <= 0 or self._step_counter % self.goal_probe_interval != 0:
            return {}
        step = max(num_steps // 2, 0)
        z_t = chain[:, step].to(dtype)
        t_batch = planner.make_timesteps(batch_size, int(planner.ddim_t[step].item()), device)
        idx_batch = planner.make_timesteps(batch_size, step, device)

        shifted = gt_goal.clone()
        shifted[:, 1] = shifted[:, 1] + self.goal_probe_shift_m

        outs = []
        for goal in (gt_goal, shifted):
            with planner.goal_context(goal):
                _, _, x0 = planner.p_mean_variance(
                    z_t, t_batch, idx_batch, vl_e, his_e, ego_e, deterministic=True
                )
            outs.append(planner.denorm_odo(x0.float()))
        shift_m = (outs[1][:, -1, :2] - outs[0][:, -1, :2]).norm(dim=-1).mean()
        return {
            "goal_effect_endpoint_shift_m": shift_m,
            "goal_effect_follow_ratio": shift_m / max(self.goal_probe_shift_m, 1e-6),
        }

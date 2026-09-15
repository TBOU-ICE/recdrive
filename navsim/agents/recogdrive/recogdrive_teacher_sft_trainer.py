"""Teacher-rollout SFT loss for the deployable GoalBridge student. Additive file.

Phase B of the teacher-rollout pipeline.  The student regresses the cached
privileged-teacher trajectory while being conditioned on *that trajectory's own
endpoint*, not on the GT endpoint.  Three consequences follow, and they are the
whole reason the stage exists:

* the ``(goal, trajectory)`` pair is self-consistent, so the mapping the student
  has to learn is deterministic and fully attainable -- no residual information
  the student cannot observe, hence none of the mode-averaging that regressing a
  GT-goal-conditioned target under an unconditioned model would cause;
* ``goal_encoder`` finally receives gradient.  In the adaln path the goal enters
  through ``conditioning = conditioning + goal_embedding``, so binding a goal is
  what puts the module in the graph at all.  Skipping it would leave the branch
  at its zero-init state and OPD would still start from a goal no-op;
* no ground-truth trajectory is consumed, so the stage is plain sequence-level
  KD from teacher outputs.  GT is loaded only to log diagnostics.

``goal_predictor`` detaches its inputs (see ``predict_goal_from_encoded``), so
``goal_loss`` updates that head alone and cannot reshape the scene encoders.
The two losses therefore touch disjoint parameter sets and their relative weight
is not a sensitive knob.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


class ReCogDriveTeacherSFTTrainer:
    """Diffusion regression onto cached teacher trajectories + goal recovery."""

    def __init__(
        self,
        goal_loss_weight: float = 1.0,
        traj_eval_interval: int = 50,
    ):
        self.goal_loss_weight = float(goal_loss_weight)
        # A full DDIM rollout is 5 extra forward passes; run it periodically so
        # the headline "is the student approaching the teacher" metric exists
        # without paying for it every step. Every rank steps the same number of
        # batches, so the interval stays aligned across DDP.
        self.traj_eval_interval = int(traj_eval_interval)
        self._step = 0
        self._last_eval: Dict[str, float] = {
            "student_fde_teacher_m": 0.0,
            "student_ade_teacher_m": 0.0,
            "student_fde_gt_m": 0.0,
            "student_ade_gt_m": 0.0,
        }

    @staticmethod
    def _encode(planner, vl_features, his_traj, ego_status, dtype):
        vl_embeds = planner.feature_encoder(vl_features.to(dtype))
        his_embeds = planner.his_traj_encoder(his_traj.to(dtype).unsqueeze(1)).repeat(
            1, planner.config.action_horizon, 1
        )
        ego_embeds = planner.ego_status_encoder(ego_status.to(dtype))
        return vl_embeds, his_embeds, ego_embeds

    @staticmethod
    def _goal_branch_norm(planner) -> torch.Tensor:
        """L2 norm of the zero-initialised adaln goal layer.

        Starts at exactly 0 for a student built from a goal-free checkpoint.
        Watching it leave zero is the only direct evidence that the goal branch
        stopped being a no-op, which is the single thing SFT must fix before OPD.
        """
        encoder = getattr(planner, "goal_encoder", None)
        if encoder is None:
            return torch.zeros(())
        return encoder.net[-1].weight.detach().float().norm()

    def compute_loss(
        self,
        student_planner,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
    ) -> BatchFeature:
        if not hasattr(student_planner, "predict_goal_from_encoded"):
            raise TypeError(
                "Teacher-rollout SFT requires PredictedGoalDiffusionPlanner as the student."
            )

        device = vl_features.device
        dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature
        teacher_traj = action_input.action.float()
        gt_traj = action_input.data.get("gt_trajectory")

        # The SFT goal is the teacher trajectory's own endpoint, which keeps the
        # conditioning and the regression target mutually consistent.
        sft_goal = teacher_traj[:, -1, :].contiguous()

        vl_embeds, his_embeds, ego_embeds = self._encode(
            student_planner, vl_features, his_traj, ego_status, dtype
        )

        # ---- goal recovery (updates goal_predictor only; inputs are detached)
        pred_goal, pred_goal_norm = student_planner.predict_goal_from_encoded(
            vl_embeds, his_embeds, ego_embeds
        )
        goal_norm_target = student_planner.norm_odo(sft_goal.unsqueeze(1)).squeeze(1)
        goal_norm_target = goal_norm_target.to(pred_goal_norm.dtype)
        use_heading = bool(
            getattr(getattr(student_planner, "goal_encoder", None), "use_heading", False)
        )
        goal_dims = 3 if use_heading else 2
        goal_loss = F.smooth_l1_loss(
            pred_goal_norm[..., :goal_dims],
            goal_norm_target[..., :goal_dims],
            reduction="mean",
        )

        # ---- diffusion regression onto the teacher trajectory
        target_actions = student_planner.norm_odo(teacher_traj.to(dtype))
        noise = torch.randn_like(target_actions)
        t_discrete = student_planner.sample_time(
            target_actions.shape[0], device=device, dtype=target_actions.dtype
        )
        noisy_actions = (
            student_planner.extract(
                student_planner.ddpm_sqrt_alphas_cumprod, t_discrete, target_actions.shape
            ) * target_actions
            + student_planner.extract(
                student_planner.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, target_actions.shape
            ) * noise
        )
        # encode_goal applies the disjoint mask/noise corruption while training,
        # so goal_encoder sees the same imperfect goals OPD will hand it.
        goal_embedding = student_planner._goal_embedding_for(
            sft_goal.to(dtype), target_actions.dtype
        )
        pred_noise = student_planner._dit_step(
            noisy_actions, t_discrete, vl_embeds, his_embeds, ego_embeds, goal_embedding
        )
        diffusion_loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")

        loss = diffusion_loss + self.goal_loss_weight * goal_loss

        # eta is unused by this objective but is a registered parameter; DDP with
        # find_unused_parameters would otherwise have to hunt for it every step.
        if hasattr(student_planner, "eta") and hasattr(student_planner.eta, "eta_logit"):
            eta_logit = student_planner.eta.eta_logit
            loss = loss + torch.nan_to_num(
                eta_logit, nan=0.0, posinf=0.0, neginf=0.0
            ).sum() * 0.0

        # ---- periodic full rollout: the headline convergence metric
        self._step += 1
        if self.traj_eval_interval > 0 and self._step % self.traj_eval_interval == 0:
            with torch.no_grad():
                was_training = student_planner.training
                student_planner.eval()  # also disables goal corruption for the probe
                try:
                    with student_planner.goal_context(pred_goal.detach().to(dtype)):
                        rollout = student_planner.get_action(
                            vl_features.to(dtype),
                            BatchFeature(data={
                                "state": torch.cat([ego_status, his_traj], dim=1).to(dtype),
                                "his_traj": his_traj.to(dtype),
                                "status_feature": ego_status.to(dtype),
                            }),
                            deterministic=True,
                        )["pred_traj"].float()
                finally:
                    if was_training:
                        student_planner.train()
                err_teacher = (rollout[..., :2] - teacher_traj[..., :2]).norm(dim=-1)
                self._last_eval["student_fde_teacher_m"] = float(err_teacher[:, -1].mean())
                self._last_eval["student_ade_teacher_m"] = float(err_teacher.mean())
                if gt_traj is not None:
                    err_gt = (rollout[..., :2] - gt_traj.float()[..., :2]).norm(dim=-1)
                    self._last_eval["student_fde_gt_m"] = float(err_gt[:, -1].mean())
                    self._last_eval["student_ade_gt_m"] = float(err_gt.mean())

        with torch.no_grad():
            goal_fde_teacher = (
                pred_goal.float()[..., :2] - sft_goal[..., :2]
            ).norm(dim=-1).mean()

        data: Dict[str, torch.Tensor] = {
            "loss": loss,
            "diffusion_loss": diffusion_loss.detach(),
            "goal_loss": goal_loss.detach(),
            "weighted_goal_loss": (self.goal_loss_weight * goal_loss).detach(),
            "goal_fde_teacher_m": goal_fde_teacher.detach(),
            # Non-zero means the goal branch left its zero-init no-op state.
            "goal_branch_weight_norm": self._goal_branch_norm(student_planner).to(device),
            "pred_goal_x_mean": pred_goal[:, 0].float().mean().detach(),
            "pred_goal_y_mean": pred_goal[:, 1].float().mean().detach(),
        }
        for key, value in self._last_eval.items():
            data[key] = torch.tensor(float(value), device=device)
        if gt_traj is not None:
            with torch.no_grad():
                teacher_err = (
                    teacher_traj[..., :2] - gt_traj.float()[..., :2]
                ).norm(dim=-1)
                data["teacher_fde_gt_m"] = teacher_err[:, -1].mean().detach()
                data["teacher_ade_gt_m"] = teacher_err.mean().detach()
        return BatchFeature(data=data)

"""GoalBridge scene-router OPD agent.

Old privileged teachers keep consuming GT goals.  The deployable student is a
PredictedGoalDiffusionPlanner: it predicts a goal from observable features and
conditions its own DDIM policy on that predicted goal at both train and test.
"""

from __future__ import annotations

from typing import Dict

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_goalbridge_distill_trainer import (
    ReCogDriveGoalBridgeDistillTrainer,
)
from navsim.agents.recogdrive.recogdrive_predicted_goal_planner import (
    PredictedGoalDiffusionPlanner,
)
from navsim.agents.recogdrive.recogdrive_scene_router_goal_agent import (
    ReCogDriveSceneRouterGoalAgent,
)


class ReCogDriveGoalBridgeAgent(ReCogDriveSceneRouterGoalAgent):
    """Predicted-goal student distilled from routed GT-goal teachers."""

    def __init__(
        self,
        *args,
        student_goal_mode: str = "adaln",
        goal_predictor_hidden_dim: int = 512,
        goal_predictor_dropout: float = 0.0,
        goal_loss_weight: float = 1.0,
        kd_weight: float = 1.0,
        anchor_weight: float = 0.15,
        anchor_checkpoint: str = "",
        recoverability_tau_m: float = 2.0,
        recoverability_floor: float = 0.05,
        kl_precision_clip: float = 25.0,
        collect_viz: bool = False,
        **kwargs,
    ):
        super().__init__(*args, collect_viz=collect_viz, **kwargs)

        # Replace the plain student created by the base agent with the deployable
        # predicted-goal student.  Transfer current base weights so this remains
        # usable even before initialize() reloads checkpoint_path.
        old_student = self.action_head
        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=old_student.config.sampling_method,
        )
        cfg.vlm_size = self.vlm_size
        new_student = PredictedGoalDiffusionPlanner(
            cfg,
            goal_mode=student_goal_mode,
            goal_sincos_dim=self.goal_sincos_dim,
            goal_hidden_dim=self.goal_hidden_dim,
            goal_use_heading=self.goal_use_heading,
            goal_predictor_hidden_dim=goal_predictor_hidden_dim,
            goal_predictor_dropout=goal_predictor_dropout,
        ).cuda()
        new_student.load_state_dict(old_student.state_dict(), strict=False)
        self.action_head = new_student
        for p in self.action_head.parameters():
            p.requires_grad = True

        student_dit = getattr(self.action_head, "model", None)
        if student_dit is not None and hasattr(student_dit, "set_adaln_bound"):
            student_dit.set_adaln_bound(self.student_adaln_bound)

        # Frozen goal-free base policy protects the student while the predicted
        # goal is still inaccurate.  Keep it outside the nn.Module tree so DDP
        # does not treat it as a trainable/unused child.
        self.__dict__["goalbridge_anchor_planner"] = None
        # Anchor is a safety regularizer, not a required part of GoalBridge.  Do
        # not waste GPU memory on a frozen extra planner when its weight is zero.
        if anchor_weight > 0.0 and anchor_checkpoint:
            self.__dict__["goalbridge_anchor_planner"] = self._build_and_load_planner(
                anchor_checkpoint, "goalbridge_anchor"
            )

        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveGoalBridgeDistillTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            smooth_weight=base.smooth_weight,
            goal_loss_weight=goal_loss_weight,
            kd_weight=kd_weight,
            anchor_weight=anchor_weight,
            recoverability_tau_m=recoverability_tau_m,
            recoverability_floor=recoverability_floor,
            kl_precision_clip=kl_precision_clip,
            collect_viz=collect_viz,
        )

        print(
            "[GoalBridge] student=PredictedGoalDiffusionPlanner "
            f"mode={student_goal_mode}, goal_w={goal_loss_weight:g}, kd_w={kd_weight:g}, "
            f"anchor_w={anchor_weight:g}, tau={recoverability_tau_m:g}m"
        )

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype
        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)
        if not self.cache_hidden_state:
            raise RuntimeError("GoalBridge expects cache_hidden_state=True.")

        last_hidden_state = features["last_hidden_state"].cuda()
        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_flat = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_flat], dim=1)

        if self.training:
            action_inputs = BatchFeature(data={
                "state": input_state.to(model_dtype),
                "his_traj": history_flat.to(model_dtype),
                "status_feature": status_feature.to(model_dtype),
                "action": targets["trajectory"].cuda().to(model_dtype),
            })
            buckets = self._buckets_for_tokens(tokens_list, last_hidden_state.size(0))
            return self.scene_router_trainer.compute_loss(
                student_planner=self.action_head,
                teacher_planners=self.teacher_planners,
                ref_planner=None,
                anchor_planner=self.goalbridge_anchor_planner,
                vl_features=last_hidden_state,
                action_input=action_inputs,
                bucket_per_sample=buckets,
            )

        # Deployable inference: predict goal from observable inputs, then plan.
        action_inputs = BatchFeature(data={
            "state": input_state.to(model_dtype),
            "his_traj": history_flat.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
        })
        pred_goal, _ = self.action_head.predict_goal(
            last_hidden_state, history_flat, status_feature
        )
        with self.action_head.goal_context(pred_goal):
            out = self.action_head.get_action(last_hidden_state, action_inputs)
        # Useful for offline debugging; compute_loss ignores this key.
        out["pred_goal"] = pred_goal.detach()
        return out

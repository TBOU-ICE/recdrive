"""Isolated GoalBridge agent with predicted-goal teacher and student."""

from __future__ import annotations

from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_agent import (
    ReCogDriveGoalBridgeGoalKDAgent,
)
from navsim.agents.recogdrive.recogdrive_goalbridge_pred_goal_kd_distill_trainer import (
    ReCogDriveGoalBridgePredGoalKDDistillTrainer,
)


class ReCogDriveGoalBridgePredGoalKDAgent(ReCogDriveGoalBridgeGoalKDAgent):
    """Use the student's predicted goal for both sides of ungated KD."""

    def __init__(
        self,
        *args,
        goal_loss_weight: float = 1.0,
        kd_weight: float = 1.0,
        kl_precision_clip: float = 25.0,
        collect_viz: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            goal_loss_weight=goal_loss_weight,
            kd_weight=kd_weight,
            kl_precision_clip=kl_precision_clip,
            collect_viz=collect_viz,
            **kwargs,
        )

        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveGoalBridgePredGoalKDDistillTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            smooth_weight=0.0,
            goal_loss_weight=goal_loss_weight,
            kd_weight=kd_weight,
            recoverability_tau_m=2.0,
            recoverability_floor=0.0,
            kl_precision_clip=kl_precision_clip,
            collect_viz=collect_viz,
        )

        print(
            "[GoalBridge-PredGoalKD] teacher_goal=pred student_goal=pred "
            f"goal_w={goal_loss_weight:g}, kd_w={kd_weight:g}; "
            "anchor=off, smooth=off, recoverability_gate=off"
        )

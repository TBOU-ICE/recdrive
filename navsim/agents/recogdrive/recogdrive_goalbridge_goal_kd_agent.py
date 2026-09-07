"""Isolated GoalBridge agent using only goal loss and ungated KD loss."""

from __future__ import annotations

from navsim.agents.recogdrive.recogdrive_goalbridge_agent import (
    ReCogDriveGoalBridgeAgent,
)
from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_distill_trainer import (
    ReCogDriveGoalBridgeGoalKDDistillTrainer,
)


class ReCogDriveGoalBridgeGoalKDAgent(ReCogDriveGoalBridgeAgent):
    """Predicted-goal student with full-strength KD on every routed sample."""

    def __init__(
        self,
        *args,
        goal_loss_weight: float = 1.0,
        kd_weight: float = 1.0,
        kl_precision_clip: float = 25.0,
        collect_viz: bool = False,
        **kwargs,
    ):
        # The parent still builds the student and privileged teachers. Passing a
        # zero anchor weight prevents loading the frozen anchor planner.
        kwargs.pop("anchor_weight", None)
        kwargs.pop("anchor_checkpoint", None)
        kwargs.pop("recoverability_tau_m", None)
        kwargs.pop("recoverability_floor", None)
        super().__init__(
            *args,
            goal_loss_weight=goal_loss_weight,
            kd_weight=kd_weight,
            anchor_weight=0.0,
            anchor_checkpoint="",
            recoverability_tau_m=2.0,
            recoverability_floor=0.0,
            kl_precision_clip=kl_precision_clip,
            collect_viz=collect_viz,
            **kwargs,
        )

        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveGoalBridgeGoalKDDistillTrainer(
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
            "[GoalBridge-GoalKD] objective=goal+ungated_reverse_kl "
            f"goal_w={goal_loss_weight:g}, kd_w={kd_weight:g}; "
            "anchor=off, smooth=off, recoverability_gate=off"
        )

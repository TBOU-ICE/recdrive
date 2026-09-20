"""Dedicated GoalBridge agent for the ungated KD-only ablation."""

from __future__ import annotations

from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_agent import (
    ReCogDriveGoalBridgeGoalKDAgent,
)
from navsim.agents.recogdrive.recogdrive_goalbridge_kd_only_distill_trainer import (
    ReCogDriveGoalBridgeKDOnlyDistillTrainer,
)


class ReCogDriveGoalBridgeKDOnlyAgent(ReCogDriveGoalBridgeGoalKDAgent):
    """Predicted-goal student optimized only by full-strength reverse KL."""

    def __init__(
        self,
        *args,
        goal_loss_weight: float = 0.0,
        kd_weight: float = 1.0,
        recoverability_floor: float = 1.0,
        kl_precision_clip: float = 25.0,
        collect_viz: bool = True,
        **kwargs,
    ):
        if float(goal_loss_weight) != 0.0:
            raise ValueError(
                "KD-only requires goal_loss_weight=0.0; use GoalKD for goal+KD."
            )
        if float(recoverability_floor) != 1.0:
            raise ValueError(
                "KD-only disables the recoverability gate; "
                "recoverability_floor must be 1.0."
            )
        kwargs.pop("recoverability_tau_m", None)
        super().__init__(
            *args,
            goal_loss_weight=0.0,
            kd_weight=kd_weight,
            kl_precision_clip=kl_precision_clip,
            collect_viz=collect_viz,
            **kwargs,
        )

        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveGoalBridgeKDOnlyDistillTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            smooth_weight=0.0,
            goal_loss_weight=0.0,
            kd_weight=kd_weight,
            recoverability_tau_m=2.0,
            recoverability_floor=1.0,
            kl_precision_clip=kl_precision_clip,
            collect_viz=collect_viz,
        )

        print(
            "[GoalBridge-KDOnly] objective=ungated_reverse_kl "
            f"kd_w={kd_weight:g}; goal=off anchor=off smooth=off; "
            "recoverability_floor=1.0"
        )

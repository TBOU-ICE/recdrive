"""Isolated GoalBridge GoalKD agent with weighted DDIM-step OPD."""

from __future__ import annotations

from typing import Sequence

from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_agent import (
    ReCogDriveGoalBridgeGoalKDAgent,
)
from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_weighted_distill_trainer import (
    ReCogDriveGoalBridgeGoalKDWeightedDistillTrainer,
)


class ReCogDriveGoalBridgeGoalKDWeightedAgent(ReCogDriveGoalBridgeGoalKDAgent):
    """GoalKD agent using explicit weights over the five DDIM steps."""

    DEFAULT_STEP_LOSS_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.00)

    def __init__(
        self,
        *args,
        step_loss_weights: Sequence[float] = DEFAULT_STEP_LOSS_WEIGHTS,
        **kwargs,
    ):
        # Consume the new option here so the original GoalKD agent does not see
        # an unknown constructor argument.
        super().__init__(*args, **kwargs)

        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveGoalBridgeGoalKDWeightedDistillTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            smooth_weight=0.0,
            goal_loss_weight=self.scene_router_trainer.goal_loss_weight,
            kd_weight=self.scene_router_trainer.kd_weight,
            recoverability_tau_m=2.0,
            recoverability_floor=0.0,
            kl_precision_clip=self.scene_router_trainer.kl_precision_clip,
            collect_viz=self.scene_router_trainer.collect_viz,
            step_loss_weights=step_loss_weights,
        )

        print(
            "[GoalBridge-GoalKD-Weighted] step_loss_weights="
            f"{self.scene_router_trainer.step_loss_weights}; "
            "objective=goal+ungated_reverse_kl; "
            "anchor=off, smooth=off, recoverability_gate=off"
        )

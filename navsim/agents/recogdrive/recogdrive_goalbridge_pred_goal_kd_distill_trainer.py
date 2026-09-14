"""GoalBridge KD where teacher and student share the predicted goal."""

from __future__ import annotations

import torch

from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_distill_trainer import (
    ReCogDriveGoalBridgeGoalKDDistillTrainer,
)


class ReCogDriveGoalBridgePredGoalKDDistillTrainer(
    ReCogDriveGoalBridgeGoalKDDistillTrainer
):
    """Condition both sides of the routed KD comparison on the student goal."""

    def _teacher_condition_goal(
        self,
        pred_goal: torch.Tensor,
        gt_goal: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        del gt_goal
        # The frozen teacher is a target network, so its condition is explicitly
        # detached while the student-side predicted goal remains differentiable.
        return pred_goal[indices].detach()

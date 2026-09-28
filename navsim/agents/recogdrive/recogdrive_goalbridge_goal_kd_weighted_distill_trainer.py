"""GoalBridge GoalKD with configurable DDIM-step weighting.

This module intentionally subclasses the existing GoalKD trainer.  The
original trainer remains unchanged; this variant only rescales the precision
inside each DDIM-step reverse-KL term.  Since the parent trainer averages the
five step losses, multiplying each step by ``5 * step_weight`` produces the
requested normalized step weights without changing the overall KD scale.
"""

from __future__ import annotations

from typing import Sequence

import torch

from navsim.agents.recogdrive.recogdrive_goalbridge_goal_kd_distill_trainer import (
    ReCogDriveGoalBridgeGoalKDDistillTrainer,
)


class ReCogDriveGoalBridgeGoalKDWeightedDistillTrainer(
    ReCogDriveGoalBridgeGoalKDDistillTrainer
):
    """GoalKD trainer with explicit normalized weights over five DDIM steps."""

    DEFAULT_STEP_LOSS_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.00)

    def __init__(
        self,
        *args,
        step_loss_weights: Sequence[float] = DEFAULT_STEP_LOSS_WEIGHTS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        weights = tuple(float(weight) for weight in step_loss_weights)
        if len(weights) != 5:
            raise ValueError(
                "step_loss_weights must contain exactly five values for the "
                f"five-step DDIM chain, got {len(weights)}"
            )
        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                f"step_loss_weights must be non-negative, got {weights}"
            )
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError("step_loss_weights must have a positive sum")

        # Normalize defensively so the configured vector controls relative
        # weighting while preserving the parent trainer's KD loss scale.
        self.step_loss_weights = tuple(weight / weight_sum for weight in weights)
        self._active_step = None

    def compute_loss(self, *args, **kwargs):
        # The parent calls _precision exactly once for each DDIM step.  Keeping
        # the cursor local to one compute_loss call makes this safe across
        # batches and leaves the original trainer untouched.
        self._active_step = 0
        try:
            return super().compute_loss(*args, **kwargs)
        finally:
            self._active_step = None

    def _precision(self, sigma2: torch.Tensor) -> torch.Tensor:
        if self._active_step is None:
            raise RuntimeError(
                "Weighted DDIM precision was requested outside compute_loss"
            )
        if self._active_step >= len(self.step_loss_weights):
            raise RuntimeError(
                "The sampled DDIM chain has more than five denoising steps; "
                f"configured weights={self.step_loss_weights}"
            )

        precision = super()._precision(sigma2)
        step_weight = self.step_loss_weights[self._active_step]
        self._active_step += 1

        # The parent computes kd_loss as mean(step_kd), i.e. sum(step_kd) / 5.
        # Multiplying by 5*w_i changes this to sum(w_i * step_kd), where the
        # configured weights sum to one.
        return precision * (len(self.step_loss_weights) * step_weight)

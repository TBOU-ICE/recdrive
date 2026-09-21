"""Dedicated goal-free-student agent for KD-only OPD."""

from __future__ import annotations

from navsim.agents.recogdrive.recogdrive_goalfree_kd_only_distill_trainer import (
    ReCogDriveGoalFreeKDOnlyDistillTrainer,
)
from navsim.agents.recogdrive.recogdrive_scene_router_goal_agent import (
    ReCogDriveSceneRouterGoalAgent,
)


class ReCogDriveGoalFreeKDOnlyAgent(ReCogDriveSceneRouterGoalAgent):
    """Goal-free student distilled from GT-goal routed teachers."""

    def __init__(
        self,
        *args,
        kd_weight: float = 1.0,
        kl_precision_clip: float = 25.0,
        collect_viz: bool = True,
        **kwargs,
    ):
        super().__init__(*args, collect_viz=collect_viz, **kwargs)
        if hasattr(self.action_head, "predict_goal_from_encoded"):
            raise TypeError(
                "Goal-free KD-only must build a plain student without a goal head."
            )

        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveGoalFreeKDOnlyDistillTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            kd_weight=kd_weight,
            kl_precision_clip=kl_precision_clip,
            collect_viz=collect_viz,
        )
        print(
            "[GoalFree-KDOnly] student_goal=none teacher_goal=GT "
            f"objective={kd_weight:g}*ungated_reverse_kl; "
            "goal_predictor=absent anchor=off smooth=off"
        )

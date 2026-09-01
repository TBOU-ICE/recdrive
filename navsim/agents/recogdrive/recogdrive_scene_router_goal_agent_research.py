"""
Research agent for the three scene-router GOAL-teacher OPD variants.

Additive file: subclasses ``ReCogDriveSceneRouterGoalAgent`` without editing it
(or any other existing file).  It keeps the goal agent's entire build path
(four goal-conditioned privileged teachers, strict goal-branch verification,
goal-free student) and only:

* swaps the goal distillation trainer for
  ``ReCogDriveGoalDistillResearchTrainer`` selected by ``distill_variant``
  (``kl`` / ``anchor`` / ``phf``); and
* for ``distill_variant='anchor'`` builds a frozen goal-FREE base planner
  (default = the student's IL init) and injects it as the manifold anchor.

The frozen anchor planner is stored via ``__dict__`` so ``nn.Module`` never
registers it as a child (same reasoning as the base agent's teachers / ExOPD
ref: frozen modules must stay out of the DDP tree to keep NCCL collectives
symmetric under ``find_unused_parameters``).
"""

from typing import Optional

from navsim.agents.recogdrive.recogdrive_goal_distill_trainer_research import (
    RESEARCH_VARIANTS,
    ReCogDriveGoalDistillResearchTrainer,
)
from navsim.agents.recogdrive.recogdrive_scene_router_goal_agent import (
    ReCogDriveSceneRouterGoalAgent,
)


class ReCogDriveSceneRouterGoalResearchAgent(ReCogDriveSceneRouterGoalAgent):
    """Goal-teacher OPD agent with a selectable research distillation variant."""

    def __init__(
        self,
        *args,
        distill_variant: str = "kl",
        anchor_weight: float = 0.3,
        phf_weight: float = 0.1,
        phf_flow_weight: float = 0.1,
        anchor_checkpoint: Optional[str] = None,
        collect_viz: bool = True,
        viz_interval_steps: int = 0,
        **kwargs,
    ):
        if distill_variant not in RESEARCH_VARIANTS:
            raise ValueError(
                f"distill_variant must be one of {RESEARCH_VARIANTS}, got {distill_variant!r}"
            )
        self.distill_variant = distill_variant

        super().__init__(
            *args,
            collect_viz=collect_viz,
            viz_interval_steps=viz_interval_steps,
            **kwargs,
        )

        # Replace the goal trainer (already built by super) with the research one,
        # preserving every knob so only the studied variable changes.
        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveGoalDistillResearchTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            smooth_weight=base.smooth_weight,
            match_target=base.match_target,
            exopd_lambda=base.exopd_lambda,
            collect_viz=base.collect_viz,
            variant=distill_variant,
            anchor_weight=anchor_weight,
            phf_weight=phf_weight,
            phf_flow_weight=phf_flow_weight,
        )

        # 方案二: build the frozen goal-free base planner used as the anchor.
        if distill_variant == "anchor":
            # Prefer an explicit anchor ckpt; otherwise reuse the student IL init
            # (self.checkpoint_path, set by ReCogDriveAgent.__init__).
            anchor_ckpt = anchor_checkpoint or getattr(self, "checkpoint_path", None)
            if not anchor_ckpt:
                raise ValueError(
                    "distill_variant='anchor' needs anchor_checkpoint (or checkpoint_path) "
                    "to build the frozen base-policy anchor."
                )
            # name must NOT start with 'teacher[' so the goal agent builds a plain
            # goal-free planner (super()._build_and_load_planner).
            anchor_planner = self._build_and_load_planner(anchor_ckpt, "anchor")
            self.__dict__["_anchor_planner"] = anchor_planner
            self.scene_router_trainer.anchor_planner = anchor_planner

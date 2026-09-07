"""
Scene-router OPD agent with GOAL-CONDITIONED (privileged) teachers.

Additive file: subclasses ``ReCogDriveSceneRouterAgent`` without editing it.
Differences from the base scene-router agent:

* the four scenario teachers are built as ``GoalCondDiffusionPlanner`` with
  ``teacher_goal_mode`` (the mode each checkpoint was TRAINED with), while the
  student and the optional ExOPD ref remain plain goal-free planners;
* teacher checkpoint loading is strict about the goal branch: the checkpoint's
  goal parameter set must match the model's exactly and every goal tensor must
  be non-zero.  Both failure modes are real: a goal_mode mismatch silently
  drops projection weights (observed PDMS collapse 0.92 -> 0.46), and a dead
  zero branch silently degrades the teacher to goal-free (the 2026.08.02
  channel/cross bug);
* the distillation trainer is swapped for the goal-aware subclass, which binds
  the GT-goal on every teacher call and logs distillation diagnostics.

The privileged goal never touches the student: it flows only through the
teacher's targets, which is what makes this distillation (internalisation)
rather than conditioning.
"""

import inspect
import os
from typing import Any, Dict

import torch

from navsim.agents.recogdrive.goal_cond import TRAINABLE_GOAL_MODES
from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_dit_scene_router_goal_distill_trainer import (
    ReCogDriveDiTSceneRouterGoalDistillTrainer,
)
from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_scene_router_agent import (
    ReCogDriveSceneRouterAgent,
    _resolve_checkpoint_path,
    _strip_action_head_prefixes,
)


class ReCogDriveSceneRouterGoalAgent(ReCogDriveSceneRouterAgent):
    """Scene-router OPD agent whose teachers consume the privileged GT goal."""

    def __init__(
        self,
        *args,
        teacher_goal_mode: str = "adaln",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        collect_viz: bool = True,
        viz_interval_steps: int = 500,
        **kwargs,
    ):
        if teacher_goal_mode not in TRAINABLE_GOAL_MODES:
            raise ValueError(
                f"teacher_goal_mode must be one of {TRAINABLE_GOAL_MODES}, "
                f"got {teacher_goal_mode!r}. Use the plain scene-router agent for "
                "goal-free teachers."
            )
        # Consumed by _build_and_load_planner, which the base __init__ calls;
        # assign before super().__init__ (plain attribute, pre-nn.Module is fine
        # because ReCogDriveAgent.__init__ runs nn.Module.__init__ first anyway).
        self.teacher_goal_mode = teacher_goal_mode
        self.goal_sincos_dim = int(goal_sincos_dim)
        self.goal_hidden_dim = int(goal_hidden_dim)
        self.goal_use_heading = bool(goal_use_heading)
        self.viz_interval_steps = int(viz_interval_steps)

        super().__init__(*args, **kwargs)

        # Swap the trainer for the goal-aware one, preserving every knob.
        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveDiTSceneRouterGoalDistillTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            smooth_weight=base.smooth_weight,
            match_target=base.match_target,
            exopd_lambda=base.exopd_lambda,
            collect_viz=collect_viz,
        )

    # --------------------------------------------------------------- builders
    def _build_and_load_planner(self, checkpoint_path_like: str, name: str):
        """Teachers become goal planners; the ExOPD ref stays goal-free."""
        if not name.startswith("teacher["):
            return super()._build_and_load_planner(checkpoint_path_like, name)

        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=self.action_head.config.sampling_method,
        )
        cfg.vlm_size = self.vlm_size
        planner = GoalCondDiffusionPlanner(
            cfg,
            goal_mode=self.teacher_goal_mode,
            goal_sincos_dim=self.goal_sincos_dim,
            goal_hidden_dim=self.goal_hidden_dim,
            goal_use_heading=self.goal_use_heading,
        ).cuda()

        load_kw: Dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        checkpoint_path = _resolve_checkpoint_path(checkpoint_path_like)
        checkpoint = torch.load(checkpoint_path, **load_kw)
        state = checkpoint.get("state_dict", checkpoint)
        stripped = _strip_action_head_prefixes(state)

        self._verify_goal_weights(planner, stripped, name, checkpoint_path)

        missing, unexpected = planner.load_state_dict(stripped, strict=False)
        print(
            f"[SceneRouter-GoalOPD] Loaded {name} (goal_mode={self.teacher_goal_mode}) "
            f"from {checkpoint_path}. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        for param in planner.parameters():
            param.requires_grad = False
        planner.eval()
        return planner

    def _verify_goal_weights(
        self,
        planner: GoalCondDiffusionPlanner,
        stripped_state: Dict[str, torch.Tensor],
        name: str,
        checkpoint_path: str,
    ) -> None:
        """A goal teacher must carry a live goal branch matching this goal_mode.

        Two silent failure modes are rejected loudly:
        * key-set mismatch -> the checkpoint was trained with a different
          goal_mode (or with none); strict=False would silently drop / leave
          zero-init the projection and wreck the supervision signal;
        * all-zero goal tensors -> a dead branch (the double-zero-init bug),
          which turns "goal distillation" into plain distillation unnoticed.
        """
        model_goal_keys = {k for k in planner.state_dict() if "goal_" in k}
        ckpt_goal_keys = {k for k in stripped_state if "goal_" in k}
        if ckpt_goal_keys != model_goal_keys:
            raise RuntimeError(
                f"{name}: goal_mode={self.teacher_goal_mode!r} does not match checkpoint "
                f"{checkpoint_path!r}. Goal keys only in checkpoint: "
                f"{sorted(ckpt_goal_keys - model_goal_keys)}; only in model: "
                f"{sorted(model_goal_keys - ckpt_goal_keys)}. Set teacher_goal_mode to the "
                "mode this teacher was trained with (goal-free teachers belong to the "
                "plain scene-router agent)."
            )
        dead = [k for k in sorted(ckpt_goal_keys) if stripped_state[k].abs().max().item() == 0.0]
        if dead:
            raise RuntimeError(
                f"{name}: checkpoint {checkpoint_path!r} has all-zero goal tensors {dead} -- "
                "the goal branch never trained (stacked zero-init deadlock). Retrain this "
                "teacher with the fixed goal_cond.py before distilling from it."
            )

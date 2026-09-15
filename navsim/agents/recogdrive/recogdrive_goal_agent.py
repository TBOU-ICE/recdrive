"""Agent wrapper for goal-conditioned (privileged) ReCogDrive teachers.

Subclasses :class:`ReCogDriveAgent` without touching it.  The only functional
differences are that the action head is a
:class:`GoalCondDiffusionPlanner`, and that the ground-truth goal point is bound
around every forward pass.

During training the goal needs no new feature or target builder: it is simply
``targets["trajectory"][:, -1, :]``, which ``TrajectoryTargetBuilder`` already
provides.  During evaluation it comes from the ``Scene``, which the PDM scoring
loop can hand over through its existing ``requires_scene`` branch (see
``run_pdm_score_recogdrive_goal.py``).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from navsim.common.dataclasses import AgentInput, Trajectory

from .goal_cond import GOAL_MODES
from .recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from .recogdrive_goal_planner import GoalCondDiffusionPlanner


class ReCogDriveGoalAgent(ReCogDriveAgent):
    """ReCogDrive agent whose planner can consume the ground-truth goal point."""

    def __init__(
        self,
        *args,
        goal_mode: str = "none",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        goal_dropout_p: float = 0.0,
        goal_noise_p: float = 0.0,
        goal_noise_std_xy: float = 0.0,
        goal_noise_std_heading: float = 0.0,
        goal_inpaint_weight: float = 1.0,
        goal_inpaint_heading: bool = False,
        strict_eval_goal: bool = True,
        **kwargs,
    ):
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}, got {goal_mode!r}")

        # Build the base agent with GRPO disabled so it does not pay for (or trip
        # over) the reference-policy load; the goal planner below redoes it.
        requested_grpo = kwargs.get("grpo", False)
        kwargs["grpo"] = False
        super().__init__(*args, **kwargs)

        self.grpo = requested_grpo
        self.goal_mode = goal_mode
        self.strict_eval_goal = bool(strict_eval_goal)
        self._eval_goal: Optional[torch.Tensor] = None

        if self.dit_type == "large":
            cfg = make_recogdrive_config(
                self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo,
                input_embedding_dim=1536, sampling_method=kwargs.get("sampling_method", "ddim"),
            )
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(
                self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo,
                input_embedding_dim=384, sampling_method=kwargs.get("sampling_method", "ddim"),
            )
        else:
            raise ValueError(f"Unknown dit_type: {self.dit_type!r}")

        cfg.vlm_size = self.vlm_size
        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint

        self.action_head = GoalCondDiffusionPlanner(
            cfg,
            goal_mode=goal_mode,
            goal_sincos_dim=goal_sincos_dim,
            goal_hidden_dim=goal_hidden_dim,
            goal_use_heading=goal_use_heading,
            goal_dropout_p=goal_dropout_p,
            goal_noise_p=goal_noise_p,
            goal_noise_std_xy=goal_noise_std_xy,
            goal_noise_std_heading=goal_noise_std_heading,
            goal_inpaint_weight=goal_inpaint_weight,
            goal_inpaint_heading=goal_inpaint_heading,
        ).cuda()

    # ------------------------------------------------------------------ helpers

    def _goal_from_targets(self, targets) -> Optional[torch.Tensor]:
        if targets is None:
            return None
        trajectory = targets.get("trajectory") if isinstance(targets, dict) else None
        if trajectory is None:
            return None
        return trajectory[:, -1, :]

    def _goal_from_scene(self, scene) -> torch.Tensor:
        poses = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        ).poses
        return torch.tensor(poses, dtype=torch.float32)[-1].unsqueeze(0)

    # ------------------------------------------------------------------ overrides

    def initialize(self) -> None:
        """Refuse to load a checkpoint whose goal branch does not fit this goal_mode.

        The base loader uses ``strict=False`` and pre-filters checkpoint keys, so
        loading e.g. a channel-trained checkpoint into an adaln-configured agent
        silently drops ``goal_channel_proj`` and routes the trained goal encoder
        into a conditioning pathway it never saw -- no error, but the PDM score
        collapses (observed: cross ckpt 0.46 / channel ckpt 0.54 under
        goal_mode=adaln, vs 0.92 for a matched adaln ckpt).  Catch it here.
        Loading a goal-FREE checkpoint into a goal agent stays allowed: that is
        the intended warm start.
        """
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
            ckpt_goal_keys = {
                (k[len("agent."):] if k.startswith("agent.") else k)
                for k in ckpt
                if "goal_" in k
            }
            model_goal_keys = {k for k in self.state_dict() if "goal_" in k}
            # A goal-free checkpoint (no goal keys at all) is a legitimate warm
            # start.  A goal checkpoint must match this agent's goal layout
            # exactly: extra ckpt keys mean the ckpt was trained with a "bigger"
            # mode (channel/cross into adaln); missing ones mean the opposite
            # (adaln into channel/cross), which leaves the zero-init projection
            # untouched and silently disables the goal.
            if ckpt_goal_keys and ckpt_goal_keys != model_goal_keys:
                raise RuntimeError(
                    f"goal_mode={self.goal_mode!r} does not match the checkpoint at "
                    f"{self.checkpoint_path!r}. Goal weights only in checkpoint: "
                    f"{sorted(ckpt_goal_keys - model_goal_keys)}; only in model: "
                    f"{sorted(model_goal_keys - ckpt_goal_keys)}. Set agent.goal_mode "
                    "to the mode the checkpoint was trained with."
                )
        super().initialize()

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        """Bind the ground-truth goal for the duration of the base forward pass."""
        goal = self._goal_from_targets(targets)
        if goal is None:
            goal = self._eval_goal

        if goal is not None:
            model_dtype = next(self.action_head.parameters()).dtype
            goal = goal.cuda().to(model_dtype)

        with self.action_head.goal_context(goal):
            return super().forward(features, targets, tokens_list)

    def compute_trajectory(self, agent_input: AgentInput, scene=None) -> Trajectory:
        """Inference. ``scene`` supplies the privileged goal point when available."""
        if scene is None and self.goal_mode != "none" and self.strict_eval_goal:
            raise RuntimeError(
                f"goal_mode={self.goal_mode!r} needs the ground-truth goal at evaluation "
                "time, but no Scene was provided. Use run_pdm_score_recogdrive_goal.py, "
                "which sets requires_scene=True. Set strict_eval_goal=false to evaluate "
                "goal-free on purpose."
            )

        self._eval_goal = self._goal_from_scene(scene) if scene is not None else None
        try:
            return super().compute_trajectory(agent_input)
        finally:
            self._eval_goal = None

"""Eval-only GoalBridge student: predicted goal from observations, then DiT plan.

This agent is intentionally separate from training
(``ReCogDriveGoalBridgeAgent``) and from the privileged teacher eval path
(``ReCogDriveGoalAgent``):

* no frozen teachers / anchors are loaded
* no Scene / GT goal is consumed
* PDMS uses the same ``run_pdm_score_recogdrive.py`` worker (requires_scene=False)

Existing ``agent=recogdrive_agent`` evals are unchanged.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Set

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.goal_cond import TRAINABLE_GOAL_MODES
from navsim.agents.recogdrive.recogdrive_agent import (
    ReCogDriveAgent,
    _resolve_checkpoint_path,
    make_recogdrive_config,
)
from navsim.agents.recogdrive.recogdrive_predicted_goal_planner import (
    PredictedGoalDiffusionPlanner,
)


def _student_goal_keys(keys) -> Set[str]:
    """Keep only student ``action_head`` goal tensors, dropping any teacher leftovers."""
    out: Set[str] = set()
    for key in keys:
        raw = key[len("agent.") :] if key.startswith("agent.") else key
        if raw.startswith("module."):
            raw = raw[len("module.") :]
        if not raw.startswith("action_head."):
            if raw.startswith("goal_") or raw.startswith("goal_predictor."):
                raw = "action_head." + raw
            else:
                continue
        if "goal_" in raw or ".goal_predictor." in raw:
            out.add(raw)
    return out


class ReCogDriveGoalBridgeEvalAgent(ReCogDriveAgent):
    """Deployable GoalBridge student used only for PDMS / onboard-style inference."""

    def __init__(
        self,
        *args,
        student_goal_mode: str = "adaln",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        goal_predictor_hidden_dim: int = 512,
        goal_predictor_dropout: float = 0.0,
        student_adaln_bound: float = 8.0,
        **kwargs,
    ):
        if student_goal_mode not in TRAINABLE_GOAL_MODES:
            raise ValueError(
                f"student_goal_mode must be one of {TRAINABLE_GOAL_MODES}, "
                f"got {student_goal_mode!r}"
            )

        kwargs["dit_distill"] = False
        kwargs["opd"] = False
        kwargs["grpo"] = False
        super().__init__(*args, **kwargs)

        self.student_goal_mode = student_goal_mode
        self.goal_sincos_dim = int(goal_sincos_dim)
        self.goal_hidden_dim = int(goal_hidden_dim)
        self.goal_use_heading = bool(goal_use_heading)
        self.student_adaln_bound = float(student_adaln_bound)

        old_student = self.action_head
        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=old_student.config.sampling_method,
        )
        cfg.vlm_size = self.vlm_size
        new_student = PredictedGoalDiffusionPlanner(
            cfg,
            goal_mode=student_goal_mode,
            goal_sincos_dim=self.goal_sincos_dim,
            goal_hidden_dim=self.goal_hidden_dim,
            goal_use_heading=self.goal_use_heading,
            goal_predictor_hidden_dim=goal_predictor_hidden_dim,
            goal_predictor_dropout=goal_predictor_dropout,
        ).cuda()
        new_student.load_state_dict(old_student.state_dict(), strict=False)
        self.action_head = new_student

        student_dit = getattr(self.action_head, "model", None)
        if student_dit is not None and hasattr(student_dit, "set_adaln_bound"):
            student_dit.set_adaln_bound(self.student_adaln_bound)

        print(
            "[GoalBridge-Eval] PredictedGoalDiffusionPlanner "
            f"mode={student_goal_mode}, adaln_bound={self.student_adaln_bound:g}. "
            "Inference uses the student's predicted goal (not GT)."
        )

    def initialize(self) -> None:
        if not self.checkpoint_path:
            raise RuntimeError("GoalBridge eval needs agent.checkpoint_path pointing at a GoalBridge student ckpt.")

        load_kw: Dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        ckpt_path = _resolve_checkpoint_path(self.checkpoint_path)
        ckpt_obj = torch.load(ckpt_path, **load_kw)
        ckpt = ckpt_obj.get("state_dict", ckpt_obj)
        ckpt_goal_keys = _student_goal_keys(ckpt.keys())
        model_goal_keys = _student_goal_keys(self.state_dict().keys())
        predictor_keys = {k for k in ckpt_goal_keys if ".goal_predictor." in k}

        if not predictor_keys:
            raise RuntimeError(
                f"{self.checkpoint_path!r} has no action_head.goal_predictor weights. "
                "This eval agent is for a GoalBridge student. Plain DiT / old Goal-OPD "
                "checkpoints should keep using agent=recogdrive_agent."
            )
        if ckpt_goal_keys and ckpt_goal_keys != model_goal_keys:
            raise RuntimeError(
                f"student_goal_mode={self.student_goal_mode!r} does not match checkpoint "
                f"{self.checkpoint_path!r}. Goal weights only in checkpoint: "
                f"{sorted(ckpt_goal_keys - model_goal_keys)}; only in model: "
                f"{sorted(model_goal_keys - ckpt_goal_keys)}."
            )

        super().initialize()

        sample_key = sorted(k for k in predictor_keys if k.endswith(".weight"))[0]
        ckpt_tensor = None
        for raw_key, value in ckpt.items():
            if sample_key in _student_goal_keys([raw_key]):
                ckpt_tensor = value
                break
        model_tensor = self.state_dict()[sample_key]
        if ckpt_tensor is None or not torch.allclose(
            model_tensor.detach().cpu().float(), ckpt_tensor.detach().cpu().float()
        ):
            raise RuntimeError(
                f"Failed to load {sample_key} from {ckpt_path!r} into the predicted-goal head."
            )

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        """Online/cached VLM encode, then predicted-goal DiT (never GT goal)."""
        if self.training:
            raise RuntimeError(
                "ReCogDriveGoalBridgeEvalAgent is eval-only. "
                "Train with ReCogDriveGoalBridgeAgent."
            )

        planner = self.action_head
        original_get_action = planner.get_action

        def _get_action_with_predicted_goal(
            vl_features,
            action_input: BatchFeature,
            init_actions: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ):
            history_flat = action_input.data["his_traj"]
            status_feature = action_input.data["status_feature"]
            pred_goal, _ = planner.predict_goal(vl_features, history_flat, status_feature)
            with planner.goal_context(pred_goal):
                out = original_get_action(
                    vl_features, action_input, init_actions, deterministic
                )
            out["pred_goal"] = pred_goal.detach()
            return out

        planner.get_action = _get_action_with_predicted_goal
        try:
            return super().forward(features, targets, tokens_list)
        finally:
            planner.get_action = original_get_action

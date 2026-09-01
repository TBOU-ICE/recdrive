"""Runtime-neutral evaluation agent for Privileged-OPD v2 Stage-3 teachers."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.common.dataclasses import AgentInput, Trajectory


def _load_goal_adapter_module():
    repo_root = Path(
        os.environ.get("PRIVILEGED_OPD_V2_ROOT", Path(__file__).resolve().parent)
    )
    source = (
        repo_root
        / "navsim"
        / "agents"
        / "recogdrive"
        / "privileged_opd_v2"
        / "goal_adapter_planner.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Privileged-OPD v2 planner source not found: {source}")
    module_name = "_privileged_opd_v2_goal_adapter_runtime"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


_goal_adapter = _load_goal_adapter_module()
PrivilegedGoalAdapterPlanner = _goal_adapter.PrivilegedGoalAdapterPlanner
extract_goal_points = _goal_adapter.extract_goal_points


def _torch_load(path: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _normal_state_key(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("agent."):
        key = key[len("agent.") :]
    return key


class ReCogDrivePrivilegedGoalAdapterEvalAgent(ReCogDriveAgent):
    """Goal-ON/OFF evaluator with token-paired diffusion noise."""

    def __init__(
        self,
        *args,
        privileged_goal: bool = True,
        strict_eval_goal: bool = True,
        goal_injection: str = "gated_cross",
        goal_point_mode: str = "final",
        goal_indices: Sequence[int] = (1, 4, 7),
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 512,
        goal_use_heading: bool = False,
        goal_adapter_heads: int = 8,
        paired_eval_noise: bool = True,
        eval_seed: int = 0,
        **kwargs,
    ):
        if bool(kwargs.get("grpo", False)):
            raise ValueError("Privileged teacher evaluation does not support grpo=True")
        kwargs["grpo"] = False
        super().__init__(*args, **kwargs)

        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=kwargs.get("sampling_method", "ddim"),
        )
        cfg.vlm_size = self.vlm_size
        planner = PrivilegedGoalAdapterPlanner(
            cfg,
            goal_injection=goal_injection,
            goal_point_mode=goal_point_mode,
            goal_indices=goal_indices,
            goal_sincos_dim=goal_sincos_dim,
            goal_hidden_dim=goal_hidden_dim,
            goal_use_heading=goal_use_heading,
            goal_adapter_heads=goal_adapter_heads,
        ).to(self.device)
        planner.load_state_dict(self.action_head.state_dict(), strict=False)
        self.action_head = planner

        self.privileged_goal = bool(privileged_goal)
        self.strict_eval_goal = bool(strict_eval_goal)
        self.paired_eval_noise = bool(paired_eval_noise)
        self.eval_seed = int(eval_seed)
        self.requires_scene = self.privileged_goal or self.paired_eval_noise
        self.goal_point_mode = str(goal_point_mode)
        self.goal_indices = tuple(int(index) for index in goal_indices)
        self._eval_goal: Optional[torch.Tensor] = None
        self._eval_noise_seed: Optional[int] = None

    def initialize(self) -> None:
        if self.checkpoint_path:
            checkpoint = _torch_load(str(self.checkpoint_path))
            state = checkpoint.get("state_dict", checkpoint)
            normalized = {_normal_state_key(key): value for key, value in state.items()}
            model_state = self.state_dict()
            expected = {
                f"action_head.{name}" for name in self.action_head.goal_parameter_names()
            }
            missing = sorted(
                key
                for key in expected
                if key not in normalized
                or key not in model_state
                or normalized[key].shape != model_state[key].shape
            )
            if missing:
                raise RuntimeError(
                    "Checkpoint does not match the requested Privileged-OPD v2 "
                    f"goal architecture ({self.action_head.goal_injection}/"
                    f"{self.goal_point_mode}); missing or shape-mismatched goal "
                    f"weights: {missing[:12]}"
                )
        super().initialize()
        self.eval()
        print(
            "[PrivGoalEval-v2] "
            f"goal={'ON' if self.privileged_goal else 'OFF'} "
            f"injection={self.action_head.goal_injection} "
            f"points={self.goal_point_mode} paired_noise={self.paired_eval_noise} "
            f"eval_seed={self.eval_seed}"
        )

    def _goal_from_targets(self, targets) -> Optional[torch.Tensor]:
        if not isinstance(targets, dict) or "trajectory" not in targets:
            return None
        return extract_goal_points(
            targets["trajectory"], self.goal_point_mode, self.goal_indices
        )

    def _goal_from_scene(self, scene) -> Optional[torch.Tensor]:
        num_poses = self._trajectory_sampling.num_poses
        start_frame_idx = scene.scene_metadata.num_history_frames - 1
        if len(scene.frames) < start_frame_idx + num_poses + 1:
            return None
        poses = scene.get_future_trajectory(num_trajectory_frames=num_poses).poses
        trajectory = torch.as_tensor(poses, dtype=torch.float32).unsqueeze(0)
        return extract_goal_points(
            trajectory, self.goal_point_mode, self.goal_indices
        )

    def _seed_from_scene(self, scene) -> int:
        metadata = scene.scene_metadata
        token = getattr(metadata, "initial_token", None)
        if not token:
            frame_idx = max(int(metadata.num_history_frames) - 1, 0)
            token = scene.frames[frame_idx].token
        digest = hashlib.sha256(
            f"{self.eval_seed}:{token}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "little") % (2**31)

    def forward(self, features, targets=None, tokens_list=None):
        if self.paired_eval_noise:
            if self._eval_noise_seed is None:
                raise RuntimeError("paired_eval_noise requires a per-scene evaluation seed")
            torch.manual_seed(self._eval_noise_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._eval_noise_seed)
        goal = self._goal_from_targets(targets)
        if goal is None:
            goal = self._eval_goal
        if not self.privileged_goal:
            goal = None
        if goal is not None:
            dtype = next(self.action_head.parameters()).dtype
            goal = goal.to(device=self.device, dtype=dtype)
        with self.action_head.goal_context(goal):
            return super().forward(features, targets, tokens_list)

    def compute_trajectory(
        self, agent_input: AgentInput, scene=None
    ) -> Trajectory:
        if self.paired_eval_noise:
            if scene is None:
                raise RuntimeError(
                    "paired_eval_noise requires Scene so a stable token seed can be derived"
                )
            self._eval_noise_seed = self._seed_from_scene(scene)
        if self.privileged_goal:
            if scene is None:
                if self.strict_eval_goal:
                    raise RuntimeError(
                        "Goal-ON evaluation requires Scene GT. Use a scorer that "
                        "passes Scene when agent.requires_scene is true."
                    )
                self._eval_goal = None
            else:
                self._eval_goal = self._goal_from_scene(scene)
        else:
            self._eval_goal = None
        try:
            return super().compute_trajectory(agent_input)
        finally:
            self._eval_goal = None
            self._eval_noise_seed = None

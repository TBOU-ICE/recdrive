"""IL agent for robust privileged-goal teacher training.

Training uses the GT trajectory endpoint as privileged goal, but the planner can
partition each batch into clean / noisy / masked-goal samples.  Validation uses
clean GT goals because goal corruption is disabled automatically in eval mode.
"""

from __future__ import annotations

from typing import Dict

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner


class ReCogDriveRobustGoalTeacherAgent(ReCogDriveAgent):
    def __init__(
        self,
        *args,
        goal_mode: str = "adaln",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        goal_dropout_p: float = 0.10,
        goal_noise_p: float = 0.20,
        goal_noise_std_xy: float = 1.0,
        goal_noise_std_heading: float = 0.10,
        **kwargs,
    ):
        kwargs["dit_distill"] = False
        kwargs["opd"] = False
        kwargs["grpo"] = False
        super().__init__(*args, **kwargs)

        old = self.action_head
        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=old.config.sampling_method,
        )
        cfg.vlm_size = self.vlm_size
        planner = GoalCondDiffusionPlanner(
            cfg,
            goal_mode=goal_mode,
            goal_sincos_dim=goal_sincos_dim,
            goal_hidden_dim=goal_hidden_dim,
            goal_use_heading=goal_use_heading,
            goal_dropout_p=goal_dropout_p,
            goal_noise_p=goal_noise_p,
            goal_noise_std_xy=goal_noise_std_xy,
            goal_noise_std_heading=goal_noise_std_heading,
        ).cuda()
        planner.load_state_dict(old.state_dict(), strict=False)
        self.action_head = planner
        print(
            "[RobustGoalTeacher] "
            f"mode={goal_mode} clean={1-goal_dropout_p-goal_noise_p:.2f} "
            f"noisy={goal_noise_p:.2f} masked={goal_dropout_p:.2f} "
            f"noise_xy={goal_noise_std_xy:g}m"
        )

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()
        if not self.cache_hidden_state:
            raise RuntimeError("Robust goal-teacher training expects cache_hidden_state=True.")
        if targets is None or "trajectory" not in targets:
            raise RuntimeError("Robust goal teacher requires trajectory targets to obtain the privileged goal.")

        model_dtype = next(self.action_head.parameters()).dtype
        history = features["history_trajectory"].cuda()
        status = features["status_feature"].cuda()
        last_hidden = features["last_hidden_state"].cuda()
        if history.ndim == 2:
            history = history.unsqueeze(0)
        if status.ndim == 1:
            status = status.unsqueeze(0)
        if last_hidden.ndim == 2:
            last_hidden = last_hidden.unsqueeze(0)
        history_flat = history.view(history.size(0), -1)
        traj = targets["trajectory"].cuda().to(model_dtype)
        goal = traj[:, -1, :]
        state = torch.cat([status, history_flat], dim=1)
        action_inputs = BatchFeature(data={
            "state": state.to(model_dtype),
            "his_traj": history_flat.to(model_dtype),
            "status_feature": status.to(model_dtype),
            "action": traj,
            "goal": goal,
        })

        last_hidden = last_hidden.to(model_dtype)
        if self.training:
            return self.action_head(last_hidden, action_inputs)
        return self.action_head.get_action(last_hidden, action_inputs)

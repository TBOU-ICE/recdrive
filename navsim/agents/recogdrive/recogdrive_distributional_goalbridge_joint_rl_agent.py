"""Joint OPD+GRPO stage for GoalBridge v2.

Use after an OPD-only expansion phase. The same deployable predicted-goal student
receives two gradients: distributional/local privileged distillation and on-policy
closed-loop proxy reward (Diffusion-GRPO). Decay opd_weight and/or increase
rl_weight across successive runs to implement Expand -> Sharpen.
"""
from __future__ import annotations

from typing import Dict

import torch
from transformers.feature_extraction_utils import BatchFeature

from .recogdrive_distributional_goalbridge_agent import ReCogDriveDistributionalGoalBridgeAgent


class ReCogDriveDistributionalGoalBridgeJointRLAgent(ReCogDriveDistributionalGoalBridgeAgent):
    def __init__(
        self,
        *args,
        opd_weight: float = 0.5,
        rl_weight: float = 1.0,
        rl_sample_time: int = 8,
        rl_bc_coeff: float = 0.05,
        rl_use_bc_loss: bool = True,
        **kwargs,
    ):
        kwargs["student_grpo"] = True
        super().__init__(*args, **kwargs)
        self.opd_weight = float(opd_weight)
        self.rl_weight = float(rl_weight)
        self.rl_sample_time = int(rl_sample_time)
        self.rl_bc_coeff = float(rl_bc_coeff)
        self.rl_use_bc_loss = bool(rl_use_bc_loss)
        print(
            f"[GoalBridge-v2-Joint] opd_w={self.opd_weight:g}, rl_w={self.rl_weight:g}, "
            f"G={self.rl_sample_time}, bc={self.rl_bc_coeff:g}"
        )

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        if not self.training:
            return super().forward(features, targets, tokens_list)

        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()
        model_dtype = next(self.action_head.parameters()).dtype
        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)
        if not self.cache_hidden_state:
            raise RuntimeError("GoalBridge joint RL expects cache_hidden_state=True.")

        last_hidden_state = features["last_hidden_state"].cuda().to(model_dtype)
        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)
        history_flat = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_flat], dim=1)
        action_inputs = BatchFeature(data={
            "state": input_state.to(model_dtype),
            "his_traj": history_flat.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
            "action": targets["trajectory"].cuda().to(model_dtype),
        })
        buckets = self._buckets_for_tokens(tokens_list, last_hidden_state.size(0))

        distill = self.scene_router_trainer.compute_loss(
            student_planner=self.action_head,
            teacher_planners=self.teacher_planners,
            ref_planner=None,
            anchor_planner=self.goalbridge_anchor_planner,
            vl_features=last_hidden_state,
            action_input=action_inputs,
            bucket_per_sample=buckets,
        )

        pred_goal, _ = self.action_head.predict_goal(last_hidden_state, history_flat, status_feature)
        with self.action_head.goal_context(pred_goal):
            rl = self.action_head.forward_grpo(
                last_hidden_state,
                action_inputs,
                tokens_list,
                sample_time=self.rl_sample_time,
                deterministic=False,
                bc_coeff=self.rl_bc_coeff,
                use_bc_loss=self.rl_use_bc_loss,
            )

        distill_loss_raw = distill.loss
        rl_loss_raw = rl.loss
        distill.loss = self.opd_weight * distill_loss_raw + self.rl_weight * rl_loss_raw
        distill["opd_loss_raw"] = distill_loss_raw.detach()
        distill["weighted_opd_loss"] = (self.opd_weight * distill_loss_raw).detach()
        distill["rl_loss_raw"] = rl_loss_raw.detach()
        distill["weighted_rl_loss"] = (self.rl_weight * rl_loss_raw).detach()
        for key in ("reward", "policy_loss", "bc_loss"):
            if key in rl:
                val = rl[key]
                distill[f"rl_{key}"] = val.detach() if isinstance(val, torch.Tensor) else val
        return distill

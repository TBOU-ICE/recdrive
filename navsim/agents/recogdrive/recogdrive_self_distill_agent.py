"""Privileged self-distillation agent: one model, two goal conditions.

There are no teacher checkpoints and no scene routing here.  A single
``PredictedGoalDiffusionPlanner`` is trained; each batch is run twice, once on
the GT endpoint (teacher) and once on its own predicted endpoint (student).
Inference is identical to the GoalBridge student, so the deployed model is
exactly the thing that was trained.
"""

from __future__ import annotations

from typing import Dict

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_predicted_goal_planner import (
    PredictedGoalDiffusionPlanner,
)
from navsim.agents.recogdrive.recogdrive_self_distill_trainer import (
    ReCogDriveSelfDistillTrainer,
)


class ReCogDriveSelfDistillAgent(ReCogDriveAgent):
    """Goal-conditioned DiT distilled from itself under privileged conditioning."""

    def __init__(
        self,
        *args,
        goal_mode: str = "adaln",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        goal_predictor_hidden_dim: int = 512,
        goal_predictor_dropout: float = 0.0,
        goal_detach_encoders: bool = False,
        il_weight: float = 1.0,
        kd_weight: float = 1.0,
        goal_loss_weight: float = 1.0,
        teacher_goal_dropout_p: float = 0.0,
        teacher_goal_noise_p: float = 0.2,
        teacher_goal_noise_std_xy: float = 1.0,
        kl_precision_clip: float = 25.0,
        min_sigma: float = 0.04,
        goal_probe_interval: int = 50,
        goal_probe_shift_m: float = 2.0,
        collect_viz: bool = False,
        viz_interval_steps: int = 0,
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
        planner = PredictedGoalDiffusionPlanner(
            cfg,
            goal_mode=goal_mode,
            goal_sincos_dim=goal_sincos_dim,
            goal_hidden_dim=goal_hidden_dim,
            goal_use_heading=goal_use_heading,
            goal_predictor_hidden_dim=goal_predictor_hidden_dim,
            goal_predictor_dropout=goal_predictor_dropout,
            goal_detach_encoders=goal_detach_encoders,
        ).cuda()
        # Stage 0 produces a GoalCondDiffusionPlanner; every tensor except the
        # goal head matches, so a non-strict load is the intended warm start.
        planner.load_state_dict(old.state_dict(), strict=False)
        self.action_head = planner
        for p in self.action_head.parameters():
            p.requires_grad = True

        self.collect_viz = bool(collect_viz)
        self.viz_interval_steps = int(viz_interval_steps)
        self.viz_output_dir = None
        # Named scene_router_trainer so the existing goal-OPD Lightning wrapper
        # finds last_viz without a second visualisation code path.
        self.scene_router_trainer = ReCogDriveSelfDistillTrainer(
            bucket_names=["self"],
            fallback_bucket="self",
            min_sigma=min_sigma,
            smooth_weight=0.0,
            il_weight=il_weight,
            kd_weight=kd_weight,
            goal_loss_weight=goal_loss_weight,
            teacher_goal_dropout_p=teacher_goal_dropout_p,
            teacher_goal_noise_p=teacher_goal_noise_p,
            teacher_goal_noise_std_xy=teacher_goal_noise_std_xy,
            recoverability_tau_m=2.0,
            recoverability_floor=0.0,
            kl_precision_clip=kl_precision_clip,
            goal_probe_interval=goal_probe_interval,
            goal_probe_shift_m=goal_probe_shift_m,
            collect_viz=collect_viz,
        )

        print(
            "[SelfDistill] one-model privileged self-distillation "
            f"mode={goal_mode} il_w={il_weight:g} kd_w={kd_weight:g} goal_w={goal_loss_weight:g} "
            f"teacher_goal noisy={teacher_goal_noise_p:g}@{teacher_goal_noise_std_xy:g}m "
            f"masked={teacher_goal_dropout_p:g} goal_detach_encoders={goal_detach_encoders}"
        )

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        del tokens_list  # self-distillation needs no bucket routing
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()
        if not self.cache_hidden_state:
            raise RuntimeError("Self-distillation expects cache_hidden_state=True.")

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
        last_hidden = last_hidden.to(model_dtype)
        history_flat = history.view(history.size(0), -1)
        state = torch.cat([status, history_flat], dim=1)

        if self.training:
            if targets is None or "trajectory" not in targets:
                raise RuntimeError("Self-distillation requires trajectory targets for the privileged goal.")
            action_inputs = BatchFeature(data={
                "state": state.to(model_dtype),
                "his_traj": history_flat.to(model_dtype),
                "status_feature": status.to(model_dtype),
                "action": targets["trajectory"].cuda().to(model_dtype),
            })
            return self.scene_router_trainer.compute_loss(
                student_planner=self.action_head,
                vl_features=last_hidden,
                action_input=action_inputs,
            )

        action_inputs = BatchFeature(data={
            "state": state.to(model_dtype),
            "his_traj": history_flat.to(model_dtype),
            "status_feature": status.to(model_dtype),
        })
        pred_goal, _ = self.action_head.predict_goal(last_hidden, history_flat, status)
        with self.action_head.goal_context(pred_goal):
            out = self.action_head.get_action(last_hidden, action_inputs)
        out["pred_goal"] = pred_goal.detach()
        return out

    def compute_loss(self, features, targets, predictions):
        # forward() already returns the full BatchFeature during training; the
        # base implementation would try to read predictions.loss off a dict.
        if self.training:
            return predictions
        return super().compute_loss(features, targets, predictions)

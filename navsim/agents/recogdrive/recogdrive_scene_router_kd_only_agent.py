"""Dedicated goal-free scene-router agent for KD-only reverse-KL OPD."""

from __future__ import annotations

from typing import Dict, Union

import torch.optim as optim
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.recogdrive.recogdrive_scene_router_agent import (
    ReCogDriveSceneRouterAgent,
)
from navsim.agents.recogdrive.recogdrive_scene_router_kd_only_distill_trainer import (
    ReCogDriveSceneRouterKDOnlyDistillTrainer,
)
from navsim.agents.recogdrive.utils.lr_scheduler import WarmupCosLR
from navsim.agents.recogdrive.utils.utils import build_from_configs


class ReCogDriveSceneRouterKDOnlyAgent(ReCogDriveSceneRouterAgent):
    """Plain student and plain routed RL teachers, trained only by reverse KL."""

    def __init__(
        self,
        *args,
        kd_weight: float = 1.0,
        kl_precision_clip: float = 25.0,
        scheduler_epochs: int = 50,
        warmup_epochs: int = 2,
        scheduler_min_lr: float = 1e-6,
        collect_viz: bool = True,
        viz_interval_steps: int = 500,
        **kwargs,
    ):
        # Hard-disable every non-KD component for this isolated experiment.
        kwargs["match_target"] = "mu"
        kwargs["exopd_ref_checkpoint"] = None
        kwargs["exopd_lambda"] = 1.0
        kwargs["scene_router_smooth_weight"] = 0.0

        self.scheduler_epochs = int(scheduler_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.scheduler_min_lr = float(scheduler_min_lr)
        self.viz_interval_steps = int(viz_interval_steps)
        if self.scheduler_epochs <= 0:
            raise ValueError("scheduler_epochs must be > 0")
        if self.warmup_epochs <= 0:
            raise ValueError("warmup_epochs must be > 0")
        if self.warmup_epochs >= self.scheduler_epochs:
            raise ValueError("warmup_epochs must be smaller than scheduler_epochs")

        super().__init__(*args, **kwargs)
        if hasattr(self.action_head, "predict_goal_from_encoded"):
            raise TypeError("Scene-router KD-only must use a goal-free student.")

        base = self.scene_router_trainer
        self.scene_router_trainer = ReCogDriveSceneRouterKDOnlyDistillTrainer(
            bucket_names=base.bucket_names,
            fallback_bucket=base.fallback_bucket,
            min_sigma=base.min_sigma,
            kd_weight=kd_weight,
            kl_precision_clip=kl_precision_clip,
            collect_viz=collect_viz,
        )
        print(
            "[SceneRouter-KDOnly] goal=none exopd=off smooth=off "
            f"objective={kd_weight:g}*reverse_kl scheduler="
            f"{self.scheduler_epochs}ep/{self.warmup_epochs}ep-warmup "
            f"viz={bool(collect_viz)}/{self.viz_interval_steps}steps"
        )

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(
            dict(
                type="AdamW",
                lr=self._lr,
                weight_decay=1e-4,
                betas=(0.9, 0.95),
            )
        )
        parameters = [
            parameter
            for parameter in self.action_head.parameters()
            if parameter.requires_grad
        ]
        optimizer = build_from_configs(optim, optimizer_cfg, params=parameters)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=self.scheduler_min_lr,
            epochs=self.scheduler_epochs,
            warmup_epochs=self.warmup_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

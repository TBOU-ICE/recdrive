"""Stage-3 lightweight privileged-teacher training for OPD v2."""
from __future__ import annotations

from typing import Dict, Sequence, Union

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.utils.lr_scheduler import WarmupCosLR
from .goal_adapter_planner import PrivilegedGoalAdapterPlanner, extract_goal_points


class ReCogDrivePrivilegedGoalAdapterTeacherAgent(ReCogDriveAgent):
    """Warm-start an old goal-free IL+RL expert and learn only a goal residual.

    ``checkpoint_path`` must point to the *goal-free* expert after its original
    IL+RL stages. The existing checkpoint loading path is reused; the new goal
    branch is absent from that checkpoint by design.
    """

    def __init__(
        self,
        *args,
        goal_injection: str = "gated_cross",
        goal_point_mode: str = "final",
        goal_indices: Sequence[int] = (1, 4, 7),
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 512,
        goal_use_heading: bool = False,
        goal_adapter_heads: int = 8,
        train_last_n_dit_blocks: int = 0,
        adapter_lr: float = 5e-5,
        backbone_lr_scale: float = 0.1,
        adapter_epochs: int = 10,
        **kwargs,
    ):
        kwargs["opd"] = False
        kwargs["dit_distill"] = False
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
        planner = PrivilegedGoalAdapterPlanner(
            cfg,
            goal_injection=goal_injection,
            goal_point_mode=goal_point_mode,
            goal_indices=goal_indices,
            goal_sincos_dim=goal_sincos_dim,
            goal_hidden_dim=goal_hidden_dim,
            goal_use_heading=goal_use_heading,
            goal_adapter_heads=goal_adapter_heads,
        ).cuda()
        planner.load_state_dict(old.state_dict(), strict=False)
        self.action_head = planner

        self.goal_point_mode = str(goal_point_mode)
        self.goal_indices = tuple(int(x) for x in goal_indices)
        self.train_last_n_dit_blocks = int(train_last_n_dit_blocks)
        self.adapter_lr = float(adapter_lr)
        self.backbone_lr_scale = float(backbone_lr_scale)
        self.adapter_epochs = int(adapter_epochs)
        self._configure_trainable_parameters()

    def initialize(self) -> None:
        # Load the old goal-free IL+RL checkpoint into all matching base weights,
        # then re-apply the freeze policy in case checkpoint restore changes none
        # of requires_grad but makes the intent explicit.
        super().initialize()
        self._configure_trainable_parameters()
        n_train = sum(p.numel() for p in self.action_head.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.action_head.parameters())
        print(
            f"[PrivGoalTeacher-v2] injection={self.action_head.goal_injection} "
            f"points={self.goal_point_mode} trainable={n_train/1e6:.2f}M/{n_all/1e6:.2f}M "
            f"last_blocks={self.train_last_n_dit_blocks}"
        )

    def _configure_trainable_parameters(self) -> None:
        for p in self.action_head.parameters():
            p.requires_grad = False
        goal_names = set(self.action_head.goal_parameter_names())
        for name, p in self.action_head.named_parameters():
            if name in goal_names:
                p.requires_grad = True
        n = self.train_last_n_dit_blocks
        if n > 0:
            blocks = self.action_head.model.transformer_blocks
            for block in blocks[max(0, len(blocks) - n):]:
                for p in block.parameters():
                    p.requires_grad = True
            # Let the output layer adapt weakly with the final blocks.
            for p in self.action_head.model.final_layer.parameters():
                p.requires_grad = True

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()
        if not self.cache_hidden_state:
            raise RuntimeError("Privileged goal-adapter teacher expects cached VLM hidden states.")
        if targets is None or "trajectory" not in targets:
            raise RuntimeError("GT trajectory is required to construct privileged goal points.")

        dtype = next(self.action_head.parameters()).dtype
        history = features["history_trajectory"].cuda()
        status = features["status_feature"].cuda()
        hidden = features["last_hidden_state"].cuda()
        if history.ndim == 2:
            history = history.unsqueeze(0)
        if status.ndim == 1:
            status = status.unsqueeze(0)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        hist_flat = history.view(history.shape[0], -1)
        traj = targets["trajectory"].cuda().to(dtype)
        goal_points = extract_goal_points(traj, self.goal_point_mode, self.goal_indices)
        action_input = BatchFeature(data={
            "state": torch.cat([status, hist_flat], dim=1).to(dtype),
            "his_traj": hist_flat.to(dtype),
            "status_feature": status.to(dtype),
            "action": traj,
            "goal_points": goal_points.to(dtype),
        })
        # Validation during teacher training intentionally stays Goal ON. Goal OFF
        # is a separate evaluation criterion, not the training validation loss.
        if self.training:
            return self.action_head(hidden.to(dtype), action_input)
        with self.action_head.goal_context(goal_points.to(dtype)):
            return self.action_head.get_action(hidden.to(dtype), action_input)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        goal_params = []
        base_params = []
        goal_names = set(self.action_head.goal_parameter_names())
        for name, p in self.action_head.named_parameters():
            if not p.requires_grad:
                continue
            (goal_params if name in goal_names else base_params).append(p)
        groups = []
        if goal_params:
            groups.append({"params": goal_params, "lr_scale": 1.0})
        if base_params:
            groups.append({"params": base_params, "lr_scale": self.backbone_lr_scale})
        if not groups:
            raise RuntimeError("No trainable privileged-teacher parameters.")
        optimizer = AdamW(groups, lr=self.adapter_lr, weight_decay=1e-4, betas=(0.9, 0.95))
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.adapter_lr,
            min_lr=1e-6,
            epochs=max(self.adapter_epochs, 2),
            warmup_epochs=1,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

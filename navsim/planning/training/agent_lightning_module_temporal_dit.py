from __future__ import annotations

from typing import Any, Dict, Tuple
from pathlib import Path
from datetime import datetime
import json

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningTemporalDiT(pl.LightningModule):
    """Lightning wrapper for temporal multi-teacher DiT distillation training.

    New file — does not touch the original AgentLightningModule / AgentLightningDiT
    or any other existing training path.
    """

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent
        self.debug_log_interval = 50
        self.debug_log_dir: Path = None
        self.debug_log_file: Path = None

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features, targets, tokens_list)
        predictions = self.agent.compute_loss(features, targets, prediction)

        if hasattr(predictions, "loss"):
            loss = predictions.loss
        else:
            loss = predictions

        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if not isinstance(predictions, torch.Tensor):
            for key, value in predictions.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value) and value.numel() == 1:
                    prog = key in {"temporal_loss", "distill_loss", "pred_traj_l1_to_teacher",
                                   "kl_il_mean", "kl_rl_mean"}
                    self.log(f"{logging_prefix}/{key}", value.detach(),
                             on_step=True, on_epoch=True, prog_bar=prog, sync_dist=True)

            if logging_prefix == "train" and getattr(self.agent, "dit_distill", False):
                self._maybe_log_debug(predictions=predictions, loss=loss)

        return loss

    @rank_zero_only
    def _maybe_log_debug(self, predictions: Dict[str, Any], loss: torch.Tensor) -> None:
        interval = int(getattr(self, "debug_log_interval", 50) or 50)
        if interval < 1:
            interval = 1
        step = int(self.global_step)
        if step % interval != 0:
            return

        if self.debug_log_dir is None:
            base_dir = Path(getattr(self, "debug_log_root", self.trainer.default_root_dir))
            self.debug_log_dir = base_dir / "log"
            self.debug_log_dir.mkdir(parents=True, exist_ok=True)
            self.debug_log_file = self.debug_log_dir / "temporal_multi_teacher_dit_opd_debug.log"

        def scalar(name: str, default: float = 0.0) -> float:
            if name not in predictions:
                return default
            v = predictions[name]
            if torch.is_tensor(v):
                if v.numel() == 0:
                    return default
                return float(v.detach().float().mean().item())
            try:
                return float(v)
            except Exception:
                return default

        payload = {
            "ts": datetime.utcnow().isoformat(),
            "step": step,
            "loss": round(float(loss.detach().float().item()), 8),
            "distill_loss": round(scalar("distill_loss"), 8),
            "kl_il_mean": round(scalar("kl_il_mean"), 8),
            "kl_rl_mean": round(scalar("kl_rl_mean"), 8),
            "temporal_loss": round(scalar("temporal_loss"), 8),
            "temporal_loss_raw": round(scalar("temporal_loss_raw"), 8),
            "temporal_pos_l1": round(scalar("temporal_pos_l1"), 8),
            "temporal_heading_l1": round(scalar("temporal_heading_l1"), 8),
            "temporal_acc_l1": round(scalar("temporal_acc_l1"), 8),
            "temporal_jerk_l1": round(scalar("temporal_jerk_l1"), 8),
            "temporal_yaw_rate_l1": round(scalar("temporal_yaw_rate_l1"), 8),
            "temporal_yaw_acc_l1": round(scalar("temporal_yaw_acc_l1"), 8),
            "pred_traj_l1_to_teacher": round(scalar("pred_traj_l1_to_teacher"), 8),
            "kl_mean": round(scalar("kl_mean"), 8),
        }
        with self.debug_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_after_backward(self) -> None:
        if not self.training:
            return
        total_norm_sq = torch.zeros(1, device=self.device)
        for p in self.agent.parameters():
            if p.grad is not None:
                grad_norm = p.grad.detach().data.norm(2)
                total_norm_sq += grad_norm * grad_norm
        total_norm = torch.sqrt(total_norm_sq)
        self.log("train/gradient_norm", total_norm, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Drop all frozen teacher weights from saved checkpoints."""
        filtered_sd = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("agent.teacher_backbone.")
            and not k.startswith("agent.teacher_action_head.")
            and not k.startswith("agent.teacher_il_action_head.")
            and not k.startswith("agent.teacher_rl_action_head.")
        }
        checkpoint["state_dict"] = filtered_sd

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningFourTeacherDiT(pl.LightningModule):
    """Lightning wrapper for fixed-weight four-teacher DiT OPD."""

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features, targets, tokens_list)
        output = self.agent.compute_loss(features, targets, prediction)

        loss = output.loss if hasattr(output, "loss") else output
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if not isinstance(output, torch.Tensor):
            scalar_keys = (
                "distill_loss",
                "smooth_loss",
                "weighted_smooth_loss",
                "transition_kl",
                "step_kl_mean",
                "step_kl_max",
                "sigma_mean",
                "chain_abs_max",
                "denoising_steps",
                "pred_traj_l1_to_teacher_mix",
                "student_pred_traj_mean",
                "student_pred_traj_std",
                "teacher_mix_pred_traj_mean",
                "teacher_mix_pred_traj_std",
                "teacher_student_pred_traj_abs_mean",
                "kl_progress_mean",
                "kl_rule_mean",
                "kl_safety_mean",
                "kl_general_mean",
                "teacher_weight_progress",
                "teacher_weight_rule",
                "teacher_weight_safety",
                "teacher_weight_general",
            )
            for key in scalar_keys:
                if key in output:
                    self.log(
                        f"{logging_prefix}/{key}",
                        output[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=key in {"distill_loss", "pred_traj_l1_to_teacher_mix"},
                        sync_dist=True,
                    )
            for key in list(output.keys()):
                if key.startswith(("kl_step_", "kl_raw_mix_step_", "sigma_step_")):
                    self.log(
                        f"{logging_prefix}/{key}",
                        output[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        filtered_sd = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("agent.teacher_action_heads.")
        }
        checkpoint["state_dict"] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()

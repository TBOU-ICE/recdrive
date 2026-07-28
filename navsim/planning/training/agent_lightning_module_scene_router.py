"""Lightning wrapper for scene-router four-teacher DiT OPD (v1). Additive file."""

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningSceneRouter(pl.LightningModule):
    """Lightning wrapper for scenario-routed multi-teacher DiT OPD."""

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], Any], logging_prefix: str) -> Tensor:
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
                "sigma_mean",
                "chain_abs_max",
                "denoising_steps",
                "exopd_lambda",
                "student_pred_traj_mean",
                "student_pred_traj_std",
            )
            for key in scalar_keys:
                if key in output:
                    self.log(
                        f"{logging_prefix}/{key}",
                        output[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=key == "distill_loss",
                        sync_dist=True,
                    )
            # These per-bucket keys are DATA-DEPENDENT: a key is only present for a
            # scenario bucket that appears in this rank's local batch. Logging them with
            # sync_dist=True issues one cross-rank all-reduce per key, and since different
            # ranks see different bucket mixes, the collectives mismatch and DDP deadlocks
            # at the first step (no traceback, training just hangs). They are diagnostics
            # only, so log them rank-locally (sync_dist=False) to avoid the hang.
            for key in list(output.keys()):
                if key.startswith(("kl_", "n_samples_")):
                    self.log(
                        f"{logging_prefix}/{key}",
                        output[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=False,
                    )

        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        filtered_sd = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("agent.teacher_planners.")
            and not k.startswith("agent.exopd_ref_planner.")
        }
        checkpoint["state_dict"] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], Any], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], Any], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()

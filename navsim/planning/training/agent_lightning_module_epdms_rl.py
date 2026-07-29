"""Lightning wrapper for the EPDMS GRPO expert training.

Same contract as ``AgentLightningDiT`` (batch = (features, targets, tokens)),
with the scalar whitelist extended for the EPDMS reward diagnostics and
NaN-safe logging (a batch may contain no SimScale samples / no pairs).
"""

from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent

SCALAR_KEYS = (
    "reward",
    "policy_loss",
    "bc_loss",
    "reward_navtrain",
    "reward_simscale",
    "det_ec",
    "active_group_frac",
)


class AgentLightningEpdmsRL(pl.LightningModule):
    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], Tuple[str, ...]], prefix: str) -> Tensor:
        features, targets, tokens = batch
        prediction = self.agent.forward(features, targets, tokens)
        predictions = self.agent.compute_loss(features, targets, prediction)

        loss = predictions.loss if hasattr(predictions, "loss") else predictions
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if not isinstance(predictions, torch.Tensor):
            for key in SCALAR_KEYS:
                if key not in predictions:
                    continue
                value = predictions[key]
                value_t = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
                if not torch.isfinite(value_t).all():
                    continue
                # Reward diagnostics are often CPU floats; NCCL sync_dist cannot reduce
                # CPU tensors. Rank-local logs are enough for these scalars.
                self.log(
                    f"{prefix}/{key}",
                    value_t.detach().float().cpu(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=key in ("reward", "det_ec"),
                    sync_dist=False,
                )
        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()

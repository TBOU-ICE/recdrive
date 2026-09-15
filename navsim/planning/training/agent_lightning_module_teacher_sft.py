"""Lightning wrapper for teacher-rollout SFT. Additive file.

Logs every scalar the SFT agent emits.  Only ``loss`` is synced across ranks:
extra ``sync_dist=True`` reductions interleaved with DDP backward have caused
SeqNum skew / ALLREDUCE timeouts on this job before, and the diagnostics are
rank-local by nature anyway.
"""

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningModule


class AgentLightningTeacherSFT(AgentLightningModule):
    """Lightning wrapper that surfaces the SFT diagnostics.

    The two acceptance metrics to watch are ``goal_encoder_wnorm`` (leaves 0 only
    if the goal branch actually receives gradient) and, when enabled,
    ``val/goal_sensitivity_m`` (proves the goal changes the plan rather than
    merely having non-zero weights).
    """

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], Any], logging_prefix: str) -> Tensor:
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features, targets, tokens_list)
        output = self.agent.compute_loss(features, targets, prediction)

        loss = output.loss if hasattr(output, "loss") else output
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if not isinstance(output, torch.Tensor):
            for key, value in output.items():
                if key == "loss" or not isinstance(value, torch.Tensor) or value.numel() != 1:
                    continue
                self.log(
                    f"{logging_prefix}/{key}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=key in ("sft_loss", "goal_fde_m"),
                    sync_dist=False,
                )

        return loss

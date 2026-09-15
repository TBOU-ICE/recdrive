"""Lightning wrapper for teacher-rollout SFT. Additive file.

Only ``loss`` is synced across ranks.  Extra ``sync_dist=True`` reductions
interleaved with the DDP backward have previously caused NCCL SeqNum skew on
this codebase, so every diagnostic below is logged rank-locally.
"""

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningModule

# Watch these two first: student_fde_teacher_m falling means the student is
# converging onto the teacher, and goal_branch_weight_norm leaving zero means the
# adaln goal branch stopped being a no-op. Both must move for SFT to have done
# anything OPD can build on.
_SFT_KEYS = (
    "diffusion_loss",
    "goal_loss",
    "weighted_goal_loss",
    "goal_fde_teacher_m",
    "goal_branch_weight_norm",
    "student_fde_teacher_m",
    "student_ade_teacher_m",
    "student_fde_gt_m",
    "student_ade_gt_m",
    "teacher_fde_gt_m",
    "teacher_ade_gt_m",
    "pred_goal_x_mean",
    "pred_goal_y_mean",
)


class AgentLightningTeacherSFT(AgentLightningModule):
    """Lightning module that logs the teacher-rollout SFT diagnostics."""

    def __init__(self, agent: AbstractAgent):
        super().__init__(agent)

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], Any],
        logging_prefix: str,
    ) -> Tensor:
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features, targets, tokens_list)
        output = self.agent.compute_loss(features, targets, prediction)

        loss = output.loss if hasattr(output, "loss") else output
        self.log(
            f"{logging_prefix}/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )

        if not isinstance(output, torch.Tensor):
            for key in _SFT_KEYS:
                if key in output:
                    self.log(
                        f"{logging_prefix}/{key}", output[key],
                        on_step=True, on_epoch=True, sync_dist=False,
                    )
        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

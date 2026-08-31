"""Lightning wrapper for Privileged-OPD v2 with component-gradient diagnostics."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor


class AgentLightningPrivilegedOPDV2(pl.LightningModule):
    def __init__(self, agent, grad_diag_interval: int = 100, grad_diag_max_params: int = 8):
        super().__init__()
        self.agent = agent
        self.grad_diag_interval = int(grad_diag_interval)
        self.grad_diag_max_params = int(grad_diag_max_params)

    @staticmethod
    def _norm(grads, device):
        acc = torch.zeros((), device=device, dtype=torch.float32)
        for g in grads:
            if g is not None:
                acc = acc + g.detach().float().pow(2).sum()
        return acc.sqrt()

    def _component_grad_diagnostics(self, out) -> None:
        if self.grad_diag_interval <= 0 or int(self.global_step) % self.grad_diag_interval != 0:
            return
        opd = out.get("_opd_component", None)
        task = out.get("_task_component", None)
        if opd is None or task is None or not task.requires_grad:
            return
        # A small deterministic subset makes this diagnostic affordable. It is a
        # scale monitor, not an optimizer operation; autograd.grad does not write
        # into .grad. All DDP ranks execute the same calls/order.
        params = [p for p in self.agent.action_head.parameters() if p.requires_grad]
        params = params[-max(self.grad_diag_max_params, 1):]
        g_opd = torch.autograd.grad(opd, params, retain_graph=True, allow_unused=True)
        g_task = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
        n_opd = self._norm(g_opd, opd.device)
        n_task = self._norm(g_task, opd.device)
        ratio = n_opd / (n_task + 1e-12)
        self.log("train/grad_opd_probe", n_opd, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/grad_task_probe", n_task, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/grad_opd_task_ratio", ratio, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], Any], prefix: str):
        features, targets, tokens = batch
        pred = self.agent.forward(features, targets, tokens)
        out = self.agent.compute_loss(features, targets, pred)
        loss = out.loss if hasattr(out, "loss") else out
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if not isinstance(out, torch.Tensor):
            for key in (
                "opd_loss", "task_loss", "goal_aux_loss", "goal_pref_loss", "smooth_loss",
                "weighted_opd_loss", "weighted_task_loss", "weighted_goal_aux_loss",
                "weighted_goal_pref_loss", "precision_norm", "sigma_mean", "x0_gap_m",
                "goal_aux_fde_m", "goal_pref_top1_fde_m", "goal_pref_teacher_gap_m", "denoising_steps", "residual_privilege",
            ):
                if key in out:
                    self.log(f"{prefix}/{key}", out[key], on_step=True, on_epoch=True,
                             prog_bar=key in ("opd_loss", "x0_gap_m"), sync_dist=True)
            if prefix == "train":
                self._component_grad_diagnostics(out)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        bad = False
        total_sq = torch.zeros((), device=self.device)
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    bad = True
                    break
                total_sq = total_sq + p.grad.detach().float().pow(2).sum()
            if bad:
                break
        if bad:
            optimizer.zero_grad(set_to_none=True)
        self.log("train/gradient_norm_total", total_sq.sqrt(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/skipped_nonfinite_grad", float(bad), on_step=True, on_epoch=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # teacher_planners/ref are kept outside the module tree, so only the
        # deployable student + optional training heads are present here.
        pass

import json
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from torch import Tensor
from typing import Any, Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features, targets, tokens_list)
        loss = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()


class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent
        self.debug_log_interval = 50
        self.debug_log_dir: Path = None
        self.debug_log_file: Path = None

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        features, targets, tokens_list = batch
        rk = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if logging_prefix == "train" and getattr(self.agent, "opd", False):
            self.agent._opd_decode_sample_this_step = rk == 0 and self.global_step % 100 == 0
        else:
            setattr(self.agent, "_opd_decode_sample_this_step", False)

        prediction = self.agent.forward(features, targets, tokens_list)
        predictions = self.agent.compute_loss(features, targets, prediction)

        # compute_loss returns either a plain tensor (IL val) or a BatchFeature (grpo/opd/dit_distill)
        if hasattr(predictions, "loss"):
            loss = predictions.loss
        else:
            loss = predictions

        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if not isinstance(predictions, torch.Tensor):
            scalar_keys = (
                "reward", "policy_loss", "bc_loss", "opd_loss", "reward_mean", "reward_weight",
                "distill_loss", "kl_mean", "sigma_mean", "chain_abs_max",
                "policy_entropy", "response_length_mean",
                "transition_kl", "step_kl_mean", "step_kl_max", "pred_traj_l1_to_teacher",
            )
            for key in scalar_keys:
                if key in predictions:
                    prog = key in ("policy_entropy", "response_length_mean")
                    self.log(f"{logging_prefix}/{key}", predictions[key],
                             on_step=True, on_epoch=True, prog_bar=prog, sync_dist=True)
            # Per-step distillation diagnostics (kl_step_i, kl_raw_step_i, sigma_step_i)
            for key in list(predictions.keys()):
                if any(key.startswith(pfx) for pfx in ("kl_step_", "kl_raw_step_", "sigma_step_")):
                    self.log(f"{logging_prefix}/{key}", predictions[key],
                             on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            if logging_prefix == "train" and self.global_step % 100 == 0:
                if "sample_response_text" in predictions and predictions["sample_response_text"]:
                    self._print_sample_response(str(predictions["sample_response_text"]))

            if logging_prefix == "train" and getattr(self.agent, "dit_distill", False):
                self._maybe_log_dit_opd_debug(predictions=predictions, loss=loss)

        return loss

    @rank_zero_only
    def _print_sample_response(self, sample_text: str) -> None:
        print(f"[train][step={self.global_step}] sample_response: {sample_text}")

    @rank_zero_only
    def _maybe_log_dit_opd_debug(self, predictions: Dict[str, Any], loss: torch.Tensor) -> None:
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
            self.debug_log_file = self.debug_log_dir / "dit_opd_debug.log"

        def _scalar(name: str, default: float = 0.0) -> float:
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

        def _vec3(name: str):
            if name not in predictions:
                return None
            v = predictions[name]
            if not torch.is_tensor(v) or v.numel() < 3:
                return None
            flat = v.detach().float().reshape(-1)
            out = [float(flat[0].item()), float(flat[1].item()), float(flat[2].item())]
            return [round(x, 6) for x in out]

        step_kls = None
        if "step_kls" in predictions and torch.is_tensor(predictions["step_kls"]):
            step_kls = [round(float(x), 8) for x in predictions["step_kls"].detach().float().cpu().tolist()]

        lr = None
        if self.trainer is not None and getattr(self.trainer, "optimizers", None):
            opt = self.trainer.optimizers[0]
            if opt.param_groups:
                lr = float(opt.param_groups[0].get("lr", 0.0))

        payload = {
            "ts": datetime.utcnow().isoformat(),
            "step": step,
            "loss": round(float(loss.detach().float().item()), 8),
            "lr": round(lr, 12) if lr is not None else None,
            "transition_kl": round(_scalar("transition_kl"), 8),
            "step_kl_mean": round(_scalar("step_kl_mean"), 8),
            "step_kl_max": round(_scalar("step_kl_max"), 8),
            "denoising_steps": round(_scalar("denoising_steps"), 4),
            "pred_traj_l1_to_teacher": round(_scalar("pred_traj_l1_to_teacher"), 8),
            "student_pred_traj_mean": round(_scalar("student_pred_traj_mean"), 8),
            "student_pred_traj_std": round(_scalar("student_pred_traj_std"), 8),
            "teacher_pred_traj_mean": round(_scalar("teacher_pred_traj_mean"), 8),
            "teacher_pred_traj_std": round(_scalar("teacher_pred_traj_std"), 8),
            "teacher_student_pred_traj_abs_mean": round(_scalar("teacher_student_pred_traj_abs_mean"), 8),
            "student_pred_traj_first_point": _vec3("student_pred_traj_first_point"),
            "teacher_pred_traj_first_point": _vec3("teacher_pred_traj_first_point"),
            "step_kls": step_kls,
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
        """
        Drop the frozen teacher backbone and teacher DiT weights from checkpoints.
        They are not needed for resuming training or inference.
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.teacher_backbone.')
            and not k.startswith('agent.teacher_action_head.')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

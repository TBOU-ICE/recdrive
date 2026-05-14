import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from torch import Tensor
from typing import Dict, Tuple,Any

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
        prediction = self.agent.forward(features,targets,tokens_list)
        #prediction = self.agent.forward(features,targets)
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
            for key in (
                "reward", "policy_loss", "bc_loss", "opd_loss", "reward_mean", "reward_weight",
                "distill_loss", "policy_entropy", "response_length_mean"
            ):
                if key in predictions:
                    prog = key in ("policy_entropy", "response_length_mean")
                    self.log(f"{logging_prefix}/{key}", predictions[key],
                             on_step=True, on_epoch=True, prog_bar=prog, sync_dist=True)

            if logging_prefix == "train" and self.global_step % 100 == 0:
                if "sample_response_text" in predictions and predictions["sample_response_text"]:
                    self._print_sample_response(str(predictions["sample_response_text"]))
        return loss

    @rank_zero_only
    def _print_sample_response(self, sample_text: str) -> None:
        print(f"[train][step={self.global_step}] sample_response: {sample_text}")

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

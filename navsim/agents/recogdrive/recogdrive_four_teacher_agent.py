from typing import Any, Dict, Optional, Union
import inspect
import os

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_dit_four_teacher_distill_trainer import (
    ReCogDriveDiTFourTeacherDistillTrainer,
)
from navsim.agents.recogdrive.utils.lr_scheduler import WarmupCosLR
from navsim.agents.recogdrive.utils.utils import build_from_configs


def _resolve_checkpoint_path(path_like: Optional[str]) -> str:
    if not path_like:
        return ""
    path = os.path.expanduser(str(path_like))
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in (
            "ReCogDrive_Diffusion_Planner_2B_RL.ckpt",
            "ReCogDrive_Diffusion_Planner_2B_IL.ckpt",
            "last.ckpt",
        ):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        candidates = []
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.endswith(".ckpt"):
                    candidates.append(os.path.join(root, filename))
        if candidates:
            candidates = sorted(candidates)
            diffusion = [c for c in candidates if "Diffusion_Planner" in os.path.basename(c)]
            return diffusion[0] if diffusion else candidates[0]
    return path


def _strip_action_head_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state.items():
        if key.startswith("agent.action_head."):
            stripped[key[len("agent.action_head."):]] = value
        elif key.startswith("action_head."):
            stripped[key[len("action_head."):]] = value
        elif key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped


class ReCogDriveFourTeacherAgent(ReCogDriveAgent):
    """ReCogDrive agent that trains the student DiT with four fixed-weight teachers."""

    def __init__(
        self,
        *args,
        teacher_dit_checkpoint_progress: Optional[str] = None,
        teacher_dit_checkpoint_rule: Optional[str] = None,
        teacher_dit_checkpoint_safety: Optional[str] = None,
        teacher_dit_checkpoint_general: Optional[str] = None,
        teacher_weight_progress: float = 0.25,
        teacher_weight_rule: float = 0.25,
        teacher_weight_safety: float = 0.25,
        teacher_weight_general: float = 0.25,
        dit_distill_min_sigma: float = 0.04,
        dit_distill_smooth_weight: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dit_four_teacher_distill = True
        self.teacher_checkpoint_paths = {
            "progress": teacher_dit_checkpoint_progress,
            "rule": teacher_dit_checkpoint_rule,
            "safety": teacher_dit_checkpoint_safety,
            "general": teacher_dit_checkpoint_general,
        }
        self.teacher_weights = {
            "progress": teacher_weight_progress,
            "rule": teacher_weight_rule,
            "safety": teacher_weight_safety,
            "general": teacher_weight_general,
        }

        missing = [name for name, path in self.teacher_checkpoint_paths.items() if not path]
        if missing:
            raise ValueError(f"Missing four-teacher checkpoint(s): {missing}")

        self.teacher_action_heads = torch.nn.ModuleDict()
        for name, path in self.teacher_checkpoint_paths.items():
            self.teacher_action_heads[name] = self._build_and_load_teacher(path, name)

        for param in self.action_head.parameters():
            param.requires_grad = True

        self.four_teacher_distill_trainer = ReCogDriveDiTFourTeacherDistillTrainer(
            teacher_weights=self.teacher_weights,
            min_sigma=dit_distill_min_sigma,
            smooth_weight=dit_distill_smooth_weight,
        )

    def _build_and_load_teacher(self, checkpoint_path_like: str, name: str) -> ReCogDriveDiffusionPlanner:
        teacher_cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=self.action_head.config.sampling_method,
        )
        teacher_cfg.vlm_size = self.vlm_size
        teacher = ReCogDriveDiffusionPlanner(teacher_cfg).cuda()

        load_kw: Dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        checkpoint_path = _resolve_checkpoint_path(checkpoint_path_like)
        checkpoint = torch.load(checkpoint_path, **load_kw)
        state = checkpoint.get("state_dict", checkpoint)
        stripped = _strip_action_head_prefixes(state)
        missing, unexpected = teacher.load_state_dict(stripped, strict=False)
        print(
            f"[FourTeacher-DiT-OPD] Loaded {name} teacher from {checkpoint_path}. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        return teacher

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype
        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()

        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if not self.cache_hidden_state:
            raise RuntimeError("Four-teacher DiT OPD expects cache_hidden_state=True for navtrain cache training.")

        last_hidden_state = features["last_hidden_state"].cuda()
        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)

        if self.training:
            action_inputs = BatchFeature(data={
                "state": input_state.to(model_dtype),
                "his_traj": history_trajectory_reshaped.to(model_dtype),
                "status_feature": status_feature.to(model_dtype),
                "action": targets["trajectory"].to(model_dtype),
            })
            return self.four_teacher_distill_trainer.compute_loss(
                student_planner=self.action_head,
                teacher_planners=self.teacher_action_heads,
                vl_features=last_hidden_state,
                action_input=action_inputs,
            )

        action_inputs = BatchFeature(data={
            "state": input_state.to(model_dtype),
            "his_traj": history_trajectory_reshaped.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
        })
        return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.training:
            return predictions
        pred = torch.nan_to_num(predictions["pred_traj"], nan=0.0, posinf=0.0, neginf=0.0)
        tgt = torch.nan_to_num(targets["trajectory"], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nn.functional.l1_loss(pred, tgt)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
        params = [p for p in self.action_head.parameters() if p.requires_grad]
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=50, warmup_epochs=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

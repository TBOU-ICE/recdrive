"""Scene-routed Privileged-OPD v2 agent (A/B/C/F/G variants)."""
from __future__ import annotations

import inspect
import json
import os
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_scene_router_agent import (
    BUCKET_NAMES,
    FALLBACK_BUCKET,
    _normalize_token,
    _resolve_checkpoint_path,
    _strip_action_head_prefixes,
)
from navsim.agents.recogdrive.utils.lr_scheduler import WarmupCosLR
from navsim.agents.recogdrive.utils.utils import build_from_configs
import torch.optim as optim

from .distill_trainer import PrivilegedOPDV2Trainer
from .goal_adapter_planner import PrivilegedGoalAdapterPlanner
from .student_heads import AuxiliaryGoalHead, GoalPreferenceHead


class ReCogDrivePrivilegedOPDV2Agent(ReCogDriveAgent):
    """Goal-free deployment student distilled from routed privileged teachers."""

    def __init__(
        self,
        *args,
        teacher_ckpt_progress_curbside_stopgo: Optional[str] = None,
        teacher_ckpt_rule_intersection: Optional[str] = None,
        teacher_ckpt_safety_dynamics_interaction: Optional[str] = None,
        teacher_ckpt_general_or_no_tag: Optional[str] = None,
        token_to_bucket_json=None,
        # teacher architecture (must match stage-3 checkpoints)
        teacher_goal_injection: str = "gated_cross",
        teacher_goal_point_mode: str = "final",
        teacher_goal_indices: Sequence[int] = (1, 4, 7),
        teacher_goal_sincos_dim: int = 128,
        teacher_goal_hidden_dim: int = 512,
        teacher_goal_use_heading: bool = False,
        teacher_goal_adapter_heads: int = 8,
        # objective
        kd_weight: float = 1.0,
        task_weight: float = 0.10,
        goal_aux_weight: float = 0.0,
        goal_pref_weight: float = 0.0,
        goal_pref_temperature_m: float = 2.0,
        goal_pref_candidate_count: int = 8,
        goal_pref_geo_weight: float = 0.25,
        residual_privilege: bool = False,
        residual_ref_checkpoint: Optional[str] = None,
        precision_clip: float = 25.0,
        scene_router_min_sigma: float = 0.04,
        scene_router_smooth_weight: float = 0.0,
        # student auxiliary heads
        goal_aux_hidden_dim: int = 512,
        goal_vocab_path: Optional[str] = None,
        goal_pref_hidden_dim: int = 512,
        goal_pref_sincos_dim: int = 128,
        # optimizer
        train_epochs: int = 30,
        student_adaln_bound: float = 8.0,
        **kwargs,
    ):
        kwargs["dit_distill"] = False
        kwargs["opd"] = False
        kwargs["grpo"] = False
        super().__init__(*args, **kwargs)

        self.teacher_goal_injection = str(teacher_goal_injection)
        self.teacher_goal_point_mode = str(teacher_goal_point_mode)
        self.teacher_goal_indices = tuple(int(x) for x in teacher_goal_indices)
        self.teacher_goal_sincos_dim = int(teacher_goal_sincos_dim)
        self.teacher_goal_hidden_dim = int(teacher_goal_hidden_dim)
        self.teacher_goal_use_heading = bool(teacher_goal_use_heading)
        self.teacher_goal_adapter_heads = int(teacher_goal_adapter_heads)
        self.residual_privilege = bool(residual_privilege)
        self.train_epochs = int(train_epochs)

        paths = {
            "progress_curbside_stopgo": teacher_ckpt_progress_curbside_stopgo,
            "rule_intersection": teacher_ckpt_rule_intersection,
            "safety_dynamics_interaction": teacher_ckpt_safety_dynamics_interaction,
            "general_or_no_tag": teacher_ckpt_general_or_no_tag,
        }
        missing = [k for k, v in paths.items() if not v]
        if missing:
            raise ValueError(f"Missing privileged teacher checkpoints: {missing}")
        self.teacher_planners = {
            name: self._build_privileged_teacher(path, name) for name, path in paths.items()
        }
        self.__dict__["residual_ref_planner"] = None
        if self.residual_privilege:
            if not residual_ref_checkpoint:
                raise ValueError("residual_privilege=True requires residual_ref_checkpoint")
            self.__dict__["residual_ref_planner"] = self._build_goal_free_ref(residual_ref_checkpoint)

        self.token_to_bucket = self._load_token_to_bucket(token_to_bucket_json)

        dim = 384 if self.dit_type == "small" else 1536
        self.student_goal_head = (
            AuxiliaryGoalHead(dim, hidden_dim=goal_aux_hidden_dim, out_dim=2)
            if goal_aux_weight > 0 else None
        )
        self.goal_preference_head = None
        self.register_buffer("candidate_goals_raw", None, persistent=False)
        if goal_pref_weight > 0:
            if not goal_vocab_path:
                raise ValueError("goal_pref_weight > 0 requires goal_vocab_path")
            vocab = self._load_goal_vocab(goal_vocab_path)
            self.candidate_goals_raw = vocab
            self.goal_preference_head = GoalPreferenceHead(
                dim, hidden_dim=goal_pref_hidden_dim, sincos_dim=goal_pref_sincos_dim
            )

        for p in self.action_head.parameters():
            p.requires_grad = True
        student_dit = getattr(self.action_head, "model", None)
        if student_dit is not None and hasattr(student_dit, "set_adaln_bound"):
            student_dit.set_adaln_bound(float(student_adaln_bound))

        self.privileged_opd_trainer = PrivilegedOPDV2Trainer(
            bucket_names=BUCKET_NAMES,
            fallback_bucket=FALLBACK_BUCKET,
            min_sigma=scene_router_min_sigma,
            smooth_weight=scene_router_smooth_weight,
            kd_weight=kd_weight,
            task_weight=task_weight,
            goal_aux_weight=goal_aux_weight,
            goal_pref_weight=goal_pref_weight,
            goal_pref_temperature_m=goal_pref_temperature_m,
            goal_pref_candidate_count=goal_pref_candidate_count,
            goal_pref_geo_weight=goal_pref_geo_weight,
            residual_privilege=self.residual_privilege,
            precision_clip=precision_clip,
        )

    def _planner_cfg(self):
        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=self.action_head.config.sampling_method,
        )
        cfg.vlm_size = self.vlm_size
        return cfg

    @staticmethod
    def _torch_load(path):
        kw: Dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kw["weights_only"] = False
        return torch.load(path, **kw)

    def _build_privileged_teacher(self, path_like: str, name: str):
        planner = PrivilegedGoalAdapterPlanner(
            self._planner_cfg(),
            goal_injection=self.teacher_goal_injection,
            goal_point_mode=self.teacher_goal_point_mode,
            goal_indices=self.teacher_goal_indices,
            goal_sincos_dim=self.teacher_goal_sincos_dim,
            goal_hidden_dim=self.teacher_goal_hidden_dim,
            goal_use_heading=self.teacher_goal_use_heading,
            goal_adapter_heads=self.teacher_goal_adapter_heads,
        ).cuda()
        path = _resolve_checkpoint_path(path_like)
        state = self._torch_load(path)
        stripped = _strip_action_head_prefixes(state.get("state_dict", state))
        model_state = planner.state_dict()
        filtered = {k: v for k, v in stripped.items() if k in model_state and model_state[k].shape == v.shape}
        missing, unexpected = planner.load_state_dict(filtered, strict=False)
        goal_names = set(planner.goal_parameter_names())
        missing_goal = sorted(k for k in goal_names if k in missing)
        if missing_goal:
            raise RuntimeError(
                f"teacher[{name}] checkpoint does not match goal architecture: missing {missing_goal[:12]}"
            )
        print(
            f"[PrivilegedOPD-v2] loaded teacher[{name}] {path}; "
            f"injection={self.teacher_goal_injection}, points={self.teacher_goal_point_mode}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        for p in planner.parameters():
            p.requires_grad = False
        planner.eval()
        return planner

    def _build_goal_free_ref(self, path_like: str):
        planner = ReCogDriveDiffusionPlanner(self._planner_cfg()).cuda()
        path = _resolve_checkpoint_path(path_like)
        state = self._torch_load(path)
        stripped = _strip_action_head_prefixes(state.get("state_dict", state))
        missing, unexpected = planner.load_state_dict(stripped, strict=False)
        print(f"[PrivilegedOPD-v2] loaded residual ref {path}; missing={len(missing)}, unexpected={len(unexpected)}")
        for p in planner.parameters():
            p.requires_grad = False
        planner.eval()
        return planner

    @staticmethod
    def _load_goal_vocab(path: str) -> torch.Tensor:
        if path.endswith(".npz"):
            data = np.load(path)
            if "goals" not in data:
                raise ValueError(f"{path}: npz must contain key 'goals'")
            arr = data["goals"]
        elif path.endswith(".npy"):
            arr = np.load(path)
        else:
            arr = np.asarray(json.loads(open(path, "r", encoding="utf-8").read()), dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"goal vocabulary must be (K,>=2), got {arr.shape}")
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
        return torch.from_numpy(arr[:, :3])

    def _load_token_to_bucket(self, json_path):
        if json_path is None:
            raise ValueError("token_to_bucket_json is required")
        paths = [json_path] if isinstance(json_path, str) else list(json_path)
        mapping = {}
        for path in paths:
            if not path or not os.path.isfile(path):
                raise ValueError(f"token_to_bucket_json not found: {path}")
            raw = json.load(open(path, "r", encoding="utf-8"))
            for token, bucket in raw.items():
                norm = _normalize_token(token)
                if norm:
                    mapping[norm] = bucket if bucket in BUCKET_NAMES else FALLBACK_BUCKET
        return mapping

    def _buckets_for_tokens(self, tokens_list, batch_size):
        if tokens_list is None:
            return [FALLBACK_BUCKET] * batch_size
        out = []
        for token in tokens_list:
            norm = _normalize_token(token)
            out.append(self.token_to_bucket.get(norm, FALLBACK_BUCKET) if norm else FALLBACK_BUCKET)
        return (out + [FALLBACK_BUCKET] * batch_size)[:batch_size]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()
        if not self.cache_hidden_state:
            raise RuntimeError("PrivilegedOPD-v2 expects cached VLM hidden states")

        dtype = next(self.action_head.parameters()).dtype
        history = features["history_trajectory"].cuda()
        status = features["status_feature"].cuda()
        hidden = features["last_hidden_state"].cuda()
        if history.ndim == 2:
            history = history.unsqueeze(0)
        if status.ndim == 1:
            status = status.unsqueeze(0)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        hist_flat = history.view(history.shape[0], -1)

        if self.training:
            if targets is None or "trajectory" not in targets:
                raise RuntimeError("OPD training requires trajectory targets")
            action_input = BatchFeature(data={
                "state": torch.cat([status, hist_flat], dim=1).to(dtype),
                "his_traj": hist_flat.to(dtype),
                "status_feature": status.to(dtype),
                "action": targets["trajectory"].cuda().to(dtype),
            })
            return self.privileged_opd_trainer.compute_loss(
                student_planner=self.action_head,
                teacher_planners=self.teacher_planners,
                ref_planner=self.residual_ref_planner,
                vl_features=hidden.to(dtype),
                action_input=action_input,
                bucket_per_sample=self._buckets_for_tokens(tokens_list, hidden.shape[0]),
                aux_goal_head=self.student_goal_head,
                goal_preference_head=self.goal_preference_head,
                candidate_goals_raw=self.candidate_goals_raw,
            )

        action_input = BatchFeature(data={
            "state": torch.cat([status, hist_flat], dim=1).to(dtype),
            "his_traj": hist_flat.to(dtype),
            "status_feature": status.to(dtype),
        })
        return self.action_head.get_action(hidden.to(dtype), action_input)

    def compute_loss(self, features, targets, predictions):
        if self.training:
            return predictions
        pred = torch.nan_to_num(predictions["pred_traj"], nan=0.0, posinf=0.0, neginf=0.0)
        tgt = torch.nan_to_num(targets["trajectory"], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nn.functional.l1_loss(pred, tgt)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        params = [p for p in self.action_head.parameters() if p.requires_grad]
        if self.student_goal_head is not None:
            params += list(self.student_goal_head.parameters())
        if self.goal_preference_head is not None:
            params += list(self.goal_preference_head.parameters())
        cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
        optimizer = build_from_configs(optim, cfg, params=params)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=max(self.train_epochs, 3),
            warmup_epochs=2,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

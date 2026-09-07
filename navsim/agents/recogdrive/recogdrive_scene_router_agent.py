"""
Scene-router four-teacher DiT OPD agent for ReCogDrive (v1).

Additive file: subclasses ``ReCogDriveAgent`` without editing it. The base agent
(with ``dit_distill=False``) builds only the trainable student DiT; here we load
four frozen scenario-expert teachers + an optional frozen ExOPD reference planner,
build a token->bucket map, and route each sample to its scenario teacher.
"""

import inspect
import json
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_dit_scene_router_distill_trainer import (
    ReCogDriveDiTSceneRouterDistillTrainer,
)
from navsim.agents.recogdrive.utils.lr_scheduler import WarmupCosLR
from navsim.agents.recogdrive.utils.utils import build_from_configs

BUCKET_NAMES = [
    "progress_curbside_stopgo",
    "rule_intersection",
    "safety_dynamics_interaction",
    "general_or_no_tag",
]
FALLBACK_BUCKET = "general_or_no_tag"


def _normalize_token(token: object) -> Optional[str]:
    if token is None:
        return None
    if isinstance(token, (bytes, bytearray)):
        token = token.hex()
    if not isinstance(token, str):
        return None
    token = token.strip().lower()
    base, sep, suffix = token.rpartition("-")
    if not (sep and suffix.isdigit() and len(suffix) == 3 and base):
        token = token.replace("-", "")
    return token or None


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


class ReCogDriveSceneRouterAgent(ReCogDriveAgent):
    """ReCogDrive agent that distills a student DiT from scenario-routed teachers."""

    def __init__(
        self,
        *args,
        teacher_ckpt_progress_curbside_stopgo: Optional[str] = None,
        teacher_ckpt_rule_intersection: Optional[str] = None,
        teacher_ckpt_safety_dynamics_interaction: Optional[str] = None,
        teacher_ckpt_general_or_no_tag: Optional[str] = None,
        token_to_bucket_json: Optional[str] = None,
        teacher_select: str = "scene_route",
        match_target: str = "x0",
        exopd_ref_checkpoint: Optional[str] = None,
        exopd_lambda: float = 1.0,
        scene_router_min_sigma: float = 0.04,
        scene_router_smooth_weight: float = 0.02,
        student_adaln_bound: float = 8.0,
        **kwargs,
    ):
        # Force dit_distill off so the base agent does NOT build IL/RL teachers.
        kwargs["dit_distill"] = False
        super().__init__(*args, **kwargs)

        if teacher_select != "scene_route":
            raise NotImplementedError(
                f"teacher_select={teacher_select!r} not supported in v1 (only 'scene_route'). "
                "pdm_best/gating are v2 and require the A2 offline table."
            )
        self.teacher_select = teacher_select
        self.exopd_lambda = float(exopd_lambda)

        teacher_ckpt_paths = {
            "progress_curbside_stopgo": teacher_ckpt_progress_curbside_stopgo,
            "rule_intersection": teacher_ckpt_rule_intersection,
            "safety_dynamics_interaction": teacher_ckpt_safety_dynamics_interaction,
            "general_or_no_tag": teacher_ckpt_general_or_no_tag,
        }
        missing = [name for name, path in teacher_ckpt_paths.items() if not path]
        if missing:
            raise ValueError(f"Missing scenario-teacher checkpoint(s): {missing}")

        # Plain dict (NOT ModuleDict): frozen teachers must stay outside the DDP
        # module tree. With ModuleDict + find_unused_parameters, the first backward
        # can desync NCCL collectives across ranks (SeqNum skew / ALLREDUCE timeout).
        teacher_planners = {}
        for name, path in teacher_ckpt_paths.items():
            teacher_planners[name] = self._build_and_load_planner(path, f"teacher[{name}]")
        self.teacher_planners = teacher_planners

        # ExOPD reference planner (frozen IL base) - only needed when extrapolating.
        # Store via __dict__ so nn.Module does not register it as a child either.
        self.exopd_ref_planner = None
        if abs(self.exopd_lambda - 1.0) > 1e-6:
            if not exopd_ref_checkpoint:
                raise ValueError("exopd_lambda != 1.0 requires exopd_ref_checkpoint (IL base).")
            self.__dict__["exopd_ref_planner"] = self._build_and_load_planner(
                exopd_ref_checkpoint, "exopd_ref"
            )

        self.token_to_bucket: Dict[str, str] = self._load_token_to_bucket(token_to_bucket_json)

        for param in self.action_head.parameters():
            param.requires_grad = True

        # Soft-bound only the trainable student DiT. Frozen teachers keep the
        # original unbounded adaLN, so OPD targets do not change.
        # S*tanh(x/S) is ~identity for |x|≪S (IL/early-OPD gates are O(1)).
        self.student_adaln_bound = float(student_adaln_bound)
        student_dit = getattr(self.action_head, "model", None)
        if student_dit is not None and hasattr(student_dit, "set_adaln_bound"):
            student_dit.set_adaln_bound(self.student_adaln_bound)
            if self.student_adaln_bound > 0:
                print(
                    f"[SceneRouter-OPD] student adaLN soft-bound={self.student_adaln_bound:g} "
                    "(teachers unbound; OPD loss unchanged)"
                )

        self.scene_router_trainer = ReCogDriveDiTSceneRouterDistillTrainer(
            bucket_names=BUCKET_NAMES,
            fallback_bucket=FALLBACK_BUCKET,
            min_sigma=scene_router_min_sigma,
            smooth_weight=scene_router_smooth_weight,
            match_target=match_target,
            exopd_lambda=self.exopd_lambda,
        )

    # --------------------------------------------------------------- builders
    def _build_and_load_planner(self, checkpoint_path_like: str, name: str) -> ReCogDriveDiffusionPlanner:
        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=self.action_head.config.sampling_method,
        )
        cfg.vlm_size = self.vlm_size
        planner = ReCogDriveDiffusionPlanner(cfg).cuda()

        load_kw: Dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        checkpoint_path = _resolve_checkpoint_path(checkpoint_path_like)
        checkpoint = torch.load(checkpoint_path, **load_kw)
        state = checkpoint.get("state_dict", checkpoint)
        stripped = _strip_action_head_prefixes(state)
        missing, unexpected = planner.load_state_dict(stripped, strict=False)
        print(
            f"[SceneRouter-OPD] Loaded {name} from {checkpoint_path}. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        for param in planner.parameters():
            param.requires_grad = False
        planner.eval()
        return planner

    def _load_token_to_bucket(self, json_path) -> Dict[str, str]:
        # Accept a single path (str) or a list of paths (e.g. navtrain + simscale rounds).
        if json_path is None:
            raise ValueError("token_to_bucket_json is required.")
        paths = [json_path] if isinstance(json_path, str) else list(json_path)
        mapping: Dict[str, str] = {}
        for path in paths:
            if not path or not os.path.isfile(path):
                raise ValueError(
                    f"token_to_bucket_json not found: {path!r}. "
                    "Expected exclusive_token_to_bucket.json ({token: bucket})."
                )
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for token, bucket in raw.items():
                norm = _normalize_token(token)
                if norm and isinstance(bucket, str):
                    mapping[norm] = bucket if bucket in BUCKET_NAMES else FALLBACK_BUCKET
        if not mapping:
            raise ValueError(f"No valid token->bucket entries parsed from {paths}")
        print(f"[SceneRouter-OPD] token->bucket map loaded: {len(mapping)} tokens from {len(paths)} file(s).")
        return mapping

    def _buckets_for_tokens(self, tokens_list, batch_size: int):
        if tokens_list is None:
            return [FALLBACK_BUCKET] * batch_size
        buckets = []
        for token in tokens_list:
            norm = _normalize_token(token)
            buckets.append(self.token_to_bucket.get(norm, FALLBACK_BUCKET) if norm else FALLBACK_BUCKET)
        if len(buckets) != batch_size:
            # pad/truncate defensively to match tensor batch
            buckets = (buckets + [FALLBACK_BUCKET] * batch_size)[:batch_size]
        return buckets

    # ---------------------------------------------------------------- forward
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
            raise RuntimeError("Scene-router OPD expects cache_hidden_state=True for navtrain cache training.")

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
            bucket_per_sample = self._buckets_for_tokens(tokens_list, last_hidden_state.size(0))
            return self.scene_router_trainer.compute_loss(
                student_planner=self.action_head,
                teacher_planners=self.teacher_planners,
                ref_planner=self.exopd_ref_planner,
                vl_features=last_hidden_state,
                action_input=action_inputs,
                bucket_per_sample=bucket_per_sample,
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
        pred = predictions["pred_traj"]
        # nan_to_num turns a dead (NaN-weight) student into the zero trajectory.
        # That is what made val/loss look like a stable 4.42 after the v4 crash
        # while train/* was already NaN. Keep the sanitizer so val does not
        # itself NaN-poison logging, but do not hide the failure.
        nonfinite = ~torch.isfinite(pred)
        if bool(nonfinite.any()):
            print(
                f"[SceneRouter-OPD] get_action pred non-finite: "
                f"frac={float(nonfinite.float().mean()):.4f} "
                f"(NaN weights or DDIM overflow — not a real L1 of 4.x)"
            )
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        tgt = torch.nan_to_num(targets["trajectory"], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nn.functional.l1_loss(pred, tgt)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
        params = [p for p in self.action_head.parameters() if p.requires_grad]
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=50, warmup_epochs=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

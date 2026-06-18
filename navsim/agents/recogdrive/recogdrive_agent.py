from typing import Any, List, Dict, Optional, Union
import inspect
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .recogdrive_features import ReCogDriveFeatureBuilder ,TrajectoryTargetBuilder
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from .recogdrive_opd_trainer import ReCogDriveOPDTrainer
from .recogdrive_dit_distill_trainer import ReCogDriveDiTDistillTrainer
from .drivevla_m0_teacher import DriveVLAM0Teacher


def _resolve_checkpoint_path(path_like: Optional[str]) -> str:
    """Resolve either a checkpoint file or a directory containing a .ckpt file."""
    if not path_like:
        return ""
    path = os.path.expanduser(str(path_like))
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = []
        preferred_names = (
            "ReCogDrive_Diffusion_Planner_2B_RL.ckpt",
            "ReCogDrive_Diffusion_Planner_2B_IL.ckpt",
            "last.ckpt",
        )
        for name in preferred_names:
            p = os.path.join(path, name)
            if os.path.isfile(p):
                return p
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.endswith(".ckpt"):
                    candidates.append(os.path.join(root, fn))
        if candidates:
            candidates = sorted(candidates)
            diffusion = [c for c in candidates if "Diffusion_Planner" in os.path.basename(c)]
            return diffusion[0] if diffusion else candidates[0]
    return path


def _candidate_state_keys(key: str) -> List[str]:
    """Return likely module keys for full-agent and standalone action-head checkpoints."""
    candidates = []
    raw = key
    if raw.startswith("module."):
        raw = raw[len("module."):]
    candidates.append(raw)
    if raw.startswith("agent."):
        candidates.append(raw[len("agent."):])
    # Standalone DiT planner checkpoints often store raw keys such as
    # ``model.*``/``feature_encoder.*``.  Inside ReCogDriveAgent the same
    # parameters live under ``action_head.*``.
    expanded = []
    for k in candidates:
        expanded.append(k)
        if not k.startswith("action_head."):
            expanded.append("action_head." + k)
    # Deduplicate while preserving order.
    out = []
    seen = set()
    for k in expanded:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _filter_checkpoint_for_agent(state: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Load either a full ReCogDriveAgent checkpoint or a standalone DiT planner checkpoint.

    The public ReCogDrive DiT files are commonly named
    ``ReCogDrive_Diffusion_Planner_*.ckpt`` and may contain raw action-head
    keys.  OPD checkpoints saved by Lightning instead contain
    ``agent.action_head.*`` keys.  This helper accepts both formats and avoids
    accidentally overwriting frozen teacher modules.
    """
    filtered = {}
    for k, v in state.items():
        for cand in _candidate_state_keys(k):
            if cand.startswith("teacher_backbone.") or cand.startswith("teacher_action_head."):
                continue
            if cand in model_state and hasattr(v, "shape") and v.shape == model_state[cand].shape:
                filtered[cand] = v
                break
    return filtered


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single', 
        vlm_type: Optional[str] = 'internvl', 
        dit_type: Optional[str] = 'small', 
        sampling_method: Optional[str] = 'ddim', 
        cache_mode: bool = False, 
        cache_hidden_state: bool = True, 
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
        vlm_size: Optional[str] = 'small', 
        train_backbone: bool = False,
        # ── OPD-specific ──────────────────────────────────────────────────────
        opd: bool = False,
        teacher_vlm_path: Optional[str] = None,
        opd_topk: int = 32,
        opd_group_size: int = 8,
        opd_max_new_tokens: int = 256,
        # ── DiT distillation (Flow-OPD KL) ────────────────────────────────────
        dit_distill: bool = False,
        teacher_dit_checkpoint: Optional[str] = None,  # backward-compatible alias for RL teacher
        teacher_dit_checkpoint_il: Optional[str] = None,
        teacher_dit_checkpoint_rl: Optional[str] = None,
        dit_distill_eps_clip: float = 0.2,            # unused, kept for config compat
        dit_distill_min_sigma: float = 0.04,
        dit_distill_normalize_advantage: bool = True, # unused, kept for config compat
        dit_distill_log_dir: str = "",
        dit_distill_log_interval: int = 50,
        dit_distill_il_weight: float = 0.75,
        dit_distill_rl_weight: float = 0.25,
        dit_distill_smooth_weight: float = 0.02,
        drivevla_teacher_checkpoint: Optional[str] = None,
        drivevla_teacher_config_path: Optional[str] = None,
        drivevla_process_weight: float = 0.5,
        drivevla_preference_weight: float = 0.5,
        drivevla_num_samples: int = 8,
        drivevla_loss_ema_decay: float = 0.99,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        self.grpo = grpo
        self.opd = opd
        self.opd_max_new_tokens = opd_max_new_tokens
        self.opd_group_size = opd_group_size
        self.dit_distill = dit_distill
        self.teacher_dit_checkpoint = teacher_dit_checkpoint
        self.teacher_dit_checkpoint_il = teacher_dit_checkpoint_il
        self.teacher_dit_checkpoint_rl = teacher_dit_checkpoint_rl
        self.drivevla_teacher_checkpoint = drivevla_teacher_checkpoint
        self.drivevla_teacher_config_path = drivevla_teacher_config_path
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device

        # ── Student VLM backbone (always online in OPD mode) ──────────────────
        if opd or (not self.cache_hidden_state and not self.cache_mode):
            print("Agent running in online VLM mode. Initializing student backbone.")
            if not vlm_path or not vlm_type:
                raise ValueError("vlm_path and vlm_type are required for online VLM mode.")
            self.backbone = RecogDriveBackbone(
                model_type=vlm_type,
                checkpoint_path=vlm_path,
                device=device,
            )
            if not self.train_backbone and not opd:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            elif opd:
                # In OPD mode: only the LLM part of the student VLM is trainable
                # ViT and MLP projector stay frozen
                for name, p in self.backbone.model.named_parameters():
                    if 'vision_model' in name or 'mlp1' in name:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True

        # ── Teacher VLM (frozen, 8B) ──────────────────────────────────────────
        self.teacher_backbone = None
        if opd:
            if not teacher_vlm_path:
                raise ValueError("teacher_vlm_path is required for OPD mode.")
            print(f"Loading teacher VLM from {teacher_vlm_path}")
            self.teacher_backbone = RecogDriveBackbone(
                model_type=vlm_type,
                checkpoint_path=teacher_vlm_path,
                device=device,
            )
            for p in self.teacher_backbone.parameters():
                p.requires_grad = False
            self.teacher_backbone.model.eval()

        if self.dit_type == "large":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=1536,sampling_method=sampling_method)
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=384,sampling_method=sampling_method)

        cfg.vlm_size = self.vlm_size

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

        if opd:
            for p in self.action_head.parameters():
                p.requires_grad = False
            self.action_head.eval()

        # ── OPD trainer ───────────────────────────────────────────────────────
        self.opd_trainer = None
        if opd:
            self.opd_trainer = ReCogDriveOPDTrainer(topk=opd_topk)

        # ── DiT distillation: dual teacher DiTs (frozen) + OPD trainer ─────────
        self.teacher_action_head = None      # backward-compatible alias: RL teacher
        self.teacher_il_action_head = None
        self.teacher_rl_action_head = None
        self.dit_distill_trainer = None
        self.drivevla_teacher = None
        if dit_distill:
            # Backward compatibility: teacher_dit_checkpoint is treated as RL teacher.
            teacher_rl_ckpt = teacher_dit_checkpoint_rl or teacher_dit_checkpoint
            teacher_il_ckpt = teacher_dit_checkpoint_il
            if not teacher_rl_ckpt or not teacher_il_ckpt:
                raise ValueError(
                    "Dual-teacher dit_distill requires teacher_dit_checkpoint_il and "
                    "teacher_dit_checkpoint_rl. For backward compatibility, "
                    "teacher_dit_checkpoint may be used as the RL teacher."
                )

            def _build_and_load_teacher(ckpt_path_like: str, name: str) -> ReCogDriveDiffusionPlanner:
                teacher_cfg = make_recogdrive_config(
                    self.dit_type, action_dim=3, action_horizon=8,
                    grpo=False,
                    input_embedding_dim=384 if self.dit_type == 'small' else 1536,
                    sampling_method=sampling_method,
                )
                teacher_cfg.vlm_size = self.vlm_size
                teacher = ReCogDriveDiffusionPlanner(teacher_cfg).cuda()

                load_kw: Dict[str, Any] = {"map_location": "cpu"}
                if "weights_only" in inspect.signature(torch.load).parameters:
                    load_kw["weights_only"] = False
                ckpt_path = _resolve_checkpoint_path(ckpt_path_like)
                ckpt = torch.load(ckpt_path, **load_kw)
                print(f"[DiT distill] Loading {name} teacher checkpoint: {ckpt_path}")
                state = ckpt.get("state_dict", ckpt)
                stripped = {}
                for k, v in state.items():
                    if k.startswith("agent.action_head."):
                        stripped[k[len("agent.action_head."):]] = v
                    elif k.startswith("action_head."):
                        stripped[k[len("action_head."):]] = v
                    else:
                        stripped[k] = v
                missing, unexpected = teacher.load_state_dict(stripped, strict=False)
                print(f"[DiT distill] {name} teacher loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                for p in teacher.parameters():
                    p.requires_grad = False
                teacher.eval()
                return teacher

            self.teacher_il_action_head = _build_and_load_teacher(teacher_il_ckpt, "IL/EC")
            self.teacher_rl_action_head = _build_and_load_teacher(teacher_rl_ckpt, "RL/PDMS")
            self.teacher_action_head = self.teacher_rl_action_head

            if drivevla_teacher_checkpoint:
                self.drivevla_teacher = DriveVLAM0Teacher(
                    checkpoint_path=drivevla_teacher_checkpoint,
                    config_path=drivevla_teacher_config_path,
                ).cuda()
                self.drivevla_teacher.eval()

            # student DiT is trainable
            for p in self.action_head.parameters():
                p.requires_grad = True

            self.dit_distill_trainer = ReCogDriveDiTDistillTrainer(
                eps_clip=dit_distill_eps_clip,
                min_sigma=dit_distill_min_sigma,
                normalize_advantage=dit_distill_normalize_advantage,
                log_dir=dit_distill_log_dir if dit_distill_log_dir else None,
                log_interval=dit_distill_log_interval,
                il_weight=dit_distill_il_weight,
                rl_weight=dit_distill_rl_weight,
                smooth_weight=dit_distill_smooth_weight,
                process_weight=drivevla_process_weight,
                preference_weight=drivevla_preference_weight,
                preference_num_samples=drivevla_num_samples,
                loss_ema_decay=drivevla_loss_ema_decay,
            )

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path:
            load_kw: Dict[str, Any] = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kw["weights_only"] = False
            ckpt_path = _resolve_checkpoint_path(self.checkpoint_path)
            ckpt_obj = torch.load(ckpt_path, **load_kw)
            print(f"[ReCogDriveAgent] Loading student checkpoint: {ckpt_path}")
            ckpt = ckpt_obj.get("state_dict", ckpt_obj)
            model_dict = self.state_dict()
            filtered_ckpt = _filter_checkpoint_for_agent(ckpt, model_dict)
            missing, unexpected = self.load_state_dict(filtered_ckpt, strict=False)
            action_loaded = sum(1 for k in filtered_ckpt if k.startswith("action_head."))
            print(
                f"[ReCogDriveAgent] Loaded {len(filtered_ckpt)} tensors "
                f"({action_loaded} action_head tensors) from student checkpoint. "
                f"Missing after partial load: {len(missing)}, Unexpected: {len(unexpected)}"
            )
            if action_loaded == 0:
                raise RuntimeError(
                    "Student checkpoint did not load any action_head tensors. "
                    "Check whether checkpoint_path points to the IL DiT planner or full ReCogDrive checkpoint."
                )

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
            cache_mode=self.cache_mode,
        )]

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

        # ── OPD branch: skip VLM forward here, forward_opd handles everything ──
        if self.training and self.opd:
            if self.backbone is None:
                raise RuntimeError("Agent is in OPD mode, but student backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1:
                image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor)

            pixel_values_list = [load_image(path) for path in image_paths]
            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()

            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                ht = history_trajectory[i]
                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(ht[j, 0].item())}, '
                    f'{format_number(ht[j, 1].item())}, '
                    f'{format_number(ht[j, 2].item())})'
                    for j in range(ht.shape[0])
                ])
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_list[i].upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            return self.forward_opd(
                pixel_values=pixel_values_cat,
                questions=questions,
                num_patches_list=num_patches_list,
            )

        # ── Non-OPD branches: need last_hidden_state ─────────────────────────
        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].cuda()
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in online VLM mode, but backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1:
                image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor)

            pixel_values_list = [load_image(path) for path in image_paths]
            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()

            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                ht = history_trajectory[i]
                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(ht[j, 0].item())}, '
                    f'{format_number(ht[j, 1].item())}, '
                    f'{format_number(ht[j, 2].item())})'
                    for j in range(ht.shape[0])
                ])
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_list[i].upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)

        if self.training and not self.grpo and not self.dit_distill:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            return self.action_head(last_hidden_state, action_inputs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
        elif self.training and self.dit_distill:
            action_inputs = BatchFeature(data={
                "state": input_state.to(model_dtype),
                "his_traj": history_trajectory_reshaped.to(model_dtype),
                "status_feature": status_feature.to(model_dtype),
                "action": targets["trajectory"].to(model_dtype),
            })
            return self.dit_distill_trainer.compute_loss(
                student_planner=self.action_head,
                teacher_il_planner=self.teacher_il_action_head,
                teacher_rl_planner=self.teacher_rl_action_head,
                drivevla_teacher=self.drivevla_teacher,
                vl_features=last_hidden_state,
                action_input=action_inputs,
            )
        else:
            action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype)})
            return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def forward_opd(
        self,
        pixel_values: torch.Tensor,   # (sum(num_patches), C, H, W)
        questions: List[str],         # len = B
        num_patches_list: List[int],  # len = B
    ):
        """
        Teacher-TopK Local Support Matching (arXiv:2603.25562 §3.2).

        For each of B inputs, generate G=opd_group_size on-policy rollouts,
        then compute KL(π̂_student || q̂_teacher) restricted to teacher Top-K
        support at every response token.
        """
        assert self.opd_trainer is not None, "opd_trainer not initialized"
        assert self.teacher_backbone is not None, "teacher_backbone not initialized"

        # Lightning.train() recursively flips all submodule training flags each step.
        # Frozen components must stay in eval to disable any dropout they may have.
        self.teacher_backbone.eval()
        self.action_head.eval()

        G = self.opd_group_size
        model_dtype = next(self.backbone.model.parameters()).dtype

        # ── Expand B inputs → B*G by repeating each sample G times ───────────
        # pixel_values is packed (sum_patches, C, H, W); split by sample then repeat.
        pv_chunks = list(torch.split(pixel_values, num_patches_list, dim=0))
        pv_expanded        = torch.cat([chunk for chunk in pv_chunks for _ in range(G)], dim=0)
        questions_expanded   = [q for q in questions   for _ in range(G)]
        num_patches_expanded = [n for n in num_patches_list for _ in range(G)]

        # ── Step 1: Build prompt embeddings and inject ViT features ───────────
        prompt_input_ids, prompt_attention_mask, _, _, _ = \
            self.backbone._build_model_inputs(
                pv_expanded, questions_expanded, num_patches_expanded
            )

        with torch.no_grad():
            vit_embeds = self.backbone.model.extract_feature(pv_expanded.to(model_dtype))
            input_embeds = self.backbone.model.language_model.get_input_embeddings()(prompt_input_ids)
            BG, N, C = input_embeds.shape
            embeds_flat = input_embeds.reshape(BG * N, C)
            ids_flat    = prompt_input_ids.reshape(BG * N)
            selected    = (ids_flat == self.backbone.img_context_token_id)
            embeds_flat[selected] = vit_embeds.reshape(-1, C).to(
                device=embeds_flat.device, dtype=embeds_flat.dtype
            )
            input_embeds = embeds_flat.reshape(BG, N, C)

            # ── Step 2: On-policy generation (top-p=0.9, temp=1.0) ────────────
            generated_ids = self.backbone.model.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.opd_max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=self.backbone.tokenizer.pad_token_id,
                eos_token_id=self.backbone.tokenizer.eos_token_id,
            )  # (B*G, T_gen)

            sample_response_text = ""
            decode_sample = bool(getattr(self, "_opd_decode_sample_this_step", False))
            if decode_sample and generated_ids.numel() > 0:
                prompt_len = prompt_input_ids.shape[1]
                completion_ids = generated_ids[0]
                if generated_ids.shape[1] >= prompt_len and torch.equal(
                    generated_ids[0, :prompt_len], prompt_input_ids[0]
                ):
                    completion_ids = generated_ids[0, prompt_len:]
                sample_response_text = self.backbone.tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True,
                ).replace("\n", " ").strip()
                if len(sample_response_text) > 300:
                    sample_response_text = sample_response_text[:300] + " ..."

        # ── Step 3: Student teacher-forcing → logits with gradients ──────────
        _, student_logits, response_mask = self.backbone.forward_with_logits(
            pixel_values=pv_expanded,
            questions=questions_expanded,
            num_patches_list=num_patches_expanded,
            generated_input_ids=generated_ids,
        )  # student_logits: (B*G, T_gen, V)

        # ── Step 4: Teacher teacher-forcing → logits (frozen) ─────────────────
        with torch.no_grad():
            _, teacher_logits, _ = self.teacher_backbone.forward_with_logits(
                pixel_values=pv_expanded,
                questions=questions_expanded,
                num_patches_list=num_patches_expanded,
                generated_input_ids=generated_ids,
            )  # teacher_logits: (B*G, T_gen, V)

        # ── Step 5: Teacher-TopK LSM KL loss ──────────────────────────────────
        out = self.opd_trainer.compute_loss(
            student_logits=student_logits.float(),
            teacher_logits=teacher_logits.float(),
            response_mask=response_mask.float(),
        )
        out["sample_response_text"] = sample_response_text
        return out

    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training and self.opd:
            return predictions  # BatchFeature returned directly from forward_opd
        elif self.training and self.dit_distill:
            return predictions  # BatchFeature returned directly from dit_distill_trainer
        elif self.training:
            return predictions.loss
        else:
            pred = torch.nan_to_num(
                predictions["pred_traj"], nan=0.0, posinf=0.0, neginf=0.0
            )
            tgt = torch.nan_to_num(
                targets["trajectory"], nan=0.0, posinf=0.0, neginf=0.0
            )
            return torch.nn.functional.l1_loss(pred, tgt)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))

        if self.opd:
            assert self.backbone is not None
            params = [p for p in self.backbone.parameters() if p.requires_grad]
        elif self.dit_distill:
            params = [p for p in self.action_head.parameters() if p.requires_grad]
        else:
            params = list(self.action_head.parameters())
            if self.backbone is not None and self.train_backbone:
                params += list(self.backbone.parameters())

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        elif self.opd:
            # OPD training: cosine decay over 20 epochs with 1-epoch warmup.
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-7, epochs=20, warmup_epochs=1)
        elif self.dit_distill:
            # DiT distill training: cosine decay over 50 epochs with 2-epoch warmup.
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=50, warmup_epochs=2)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config

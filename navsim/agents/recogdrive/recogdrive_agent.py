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
        opd_norm_to_one: bool = True,
        opd_reward_weight_mode: str = 'normalize',
        opd_use_reward_weighting: bool = True,
        opd_max_new_tokens: int = 256,
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
            self.opd_trainer = ReCogDriveOPDTrainer(
                metric_cache_path=metric_cache_path,
                topk=opd_topk,
                norm_to_one=opd_norm_to_one,
                reward_weight_mode=opd_reward_weight_mode,
                use_reward_weighting=opd_use_reward_weighting,
            )

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path:
            load_kw: Dict[str, Any] = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kw["weights_only"] = False
            ckpt = torch.load(self.checkpoint_path, **load_kw)["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in ckpt.items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            self.load_state_dict(filtered_ckpt, strict=False)

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

            status_feature = features["status_feature"].cuda()
            if status_feature.ndim == 1:
                status_feature = status_feature.unsqueeze(0)

            return self.forward_opd(
                pixel_values=pixel_values_cat,
                questions=questions,
                num_patches_list=num_patches_list,
                history_trajectory=history_trajectory,
                status_feature=status_feature,
                tokens_list=tokens_list,
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

        if self.training and not self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            return self.action_head(last_hidden_state, action_inputs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
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
        pixel_values: torch.Tensor,
        questions: List[str],
        num_patches_list: List[int],
        history_trajectory: torch.Tensor,
        status_feature: torch.Tensor,
        tokens_list: List[str],
    ):
        """
        OPD forward pass:
        1. Student VLM generates tokens on-policy (top-p sampling).
        2. Both student and teacher VLM run teacher-forcing on those tokens → logits.
        3. Compute Teacher-TopK KL loss.
        4. DiT (frozen) converts student hidden states → trajectory → PDM score.
        5. Return reward-weighted OPD loss.
        """
        assert self.opd_trainer is not None, "opd_trainer not initialized"
        assert self.teacher_backbone is not None, "teacher_backbone not initialized"

        model_dtype = next(self.backbone.model.parameters()).dtype
        device = pixel_values.device

        # ── Step 1: Student VLM on-policy generation ──────────────────────────
        # Reuse _build_model_inputs to avoid duplicating tokenization logic.
        prompt_input_ids, prompt_attention_mask, _, _, _ = \
            self.backbone._build_model_inputs(pixel_values, questions, num_patches_list)

        # Generate student tokens (on-policy, top-p=0.9)
        with torch.no_grad():
            vit_embeds = self.backbone.model.extract_feature(pixel_values.to(model_dtype))
            input_embeds = self.backbone.model.language_model.get_input_embeddings()(prompt_input_ids)
            B, N, C = input_embeds.shape
            input_embeds_flat = input_embeds.reshape(B * N, C)
            ids_flat = prompt_input_ids.reshape(B * N)
            selected = (ids_flat == self.backbone.img_context_token_id)
            input_embeds_flat[selected] = vit_embeds.reshape(-1, C).to(
                device=input_embeds_flat.device, dtype=input_embeds_flat.dtype
            )
            input_embeds = input_embeds_flat.reshape(B, N, C)

            generated_ids = self.backbone.model.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.opd_max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=self.backbone.tokenizer.pad_token_id,
                eos_token_id=self.backbone.tokenizer.eos_token_id,
            )  # (B, T_gen)

        # ── Step 2: Student logits (teacher-forcing on generated tokens) ───────
        # output also contains hidden_states, reused in Step 4 to avoid a 3rd VLM forward
        student_output, student_logits, response_mask = self.backbone.forward_with_logits(
            pixel_values=pixel_values,
            questions=questions,
            num_patches_list=num_patches_list,
            generated_input_ids=generated_ids,
        )  # student_logits: (B, T_gen, V)
        # Detach hidden_states immediately to free the (2800+T_gen) activation graph;
        # only student_logits needs gradients for the KL loss.
        student_hidden = student_output.hidden_states[-1].detach()

        # ── Step 3: Teacher logits (frozen, same generated tokens) ────────────
        with torch.no_grad():
            _, teacher_logits, _ = self.teacher_backbone.forward_with_logits(
                pixel_values=pixel_values,
                questions=questions,
                num_patches_list=num_patches_list,
                generated_input_ids=generated_ids,
            )  # teacher_logits: (B, T_gen, V)

        # ── Step 4: DiT trajectory (frozen, uses student hidden states from Step 2) ───────
        # Reuse student_hidden (already detached) — no extra VLM forward needed.
        # prompt_len is derived dynamically from generated_ids offset so it stays
        # correct if _build_model_inputs max_length ever changes.
        with torch.no_grad():
            prompt_len = student_hidden.size(1) - generated_ids.size(1)
            last_hidden_state = student_hidden[:, :prompt_len, :].to(model_dtype)

            history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
            input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
            action_inputs = BatchFeature(data={
                "state": input_state.to(model_dtype),
                "his_traj": history_trajectory_reshaped.to(model_dtype),
                "status_feature": status_feature.to(model_dtype),
            })
            traj_output = self.action_head.get_action(last_hidden_state, action_inputs)
            pred_traj = traj_output["pred_traj"]  # (B, H, 3) denormalized

        # ── Step 5: Reward-weighted OPD loss ──────────────────────────────────
        result = self.opd_trainer.compute_loss(
            student_logits=student_logits.float(),
            teacher_logits=teacher_logits.float(),
            response_mask=response_mask.float(),
            pred_traj=pred_traj,
            tokens_list=list(tokens_list),
        )
        return result

    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training and self.opd:
            return predictions  # BatchFeature returned directly from forward_opd
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
        else:
            params = list(self.action_head.parameters())
            if self.backbone is not None and self.train_backbone:
                params += list(self.backbone.parameters())

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
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

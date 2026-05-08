# DiT Distillation trainer for ReCogDrive.
#
# Loss = w(PDM_score) * (lambda_out * L_out + lambda_feat * L_feat + lambda_traj * L_traj)
#
# L_out:  student/teacher prediction matching on student on-policy denoising chain
# L_feat: student/teacher intermediate DiT token feature matching on the same chain
# L_traj: student/teacher final denoised trajectory matching

import logging
import lzma
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)


def _sanitize_pred_trajectory_np(poses: np.ndarray) -> np.ndarray:
    out = np.asarray(poses, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if out.shape[-1] >= 3:
        out[..., 2] = np.arctan2(np.sin(out[..., 2]), np.cos(out[..., 2]))
    return out.astype(np.float32)


def compute_pdm_rewards(
    pred_traj: torch.Tensor,
    tokens_list: List[str],
    metric_cache_loader: MetricCacheLoader,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> torch.Tensor:
    pred_np = pred_traj.detach().float().cpu().numpy()
    pred_np = np.stack([_sanitize_pred_trajectory_np(p) for p in pred_np], axis=0)
    unique_tokens = set(tokens_list)
    cache_dict = {}
    for token in unique_tokens:
        path = metric_cache_loader.metric_cache_paths[token]
        with lzma.open(path, "rb") as f:
            cache_dict[token] = pickle.load(f)

    rewards = []
    for i, token in enumerate(tokens_list):
        traj = Trajectory(pred_np[i])
        try:
            result = pdm_score(
                metric_cache=cache_dict[token],
                model_trajectory=traj,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            rewards.append(asdict(result)["score"])
        except (AssertionError, ValueError, RuntimeError) as e:
            logger.warning("PDM scoring failed for sample %s (token=%s): %s", i, token, e)
            rewards.append(0.0)

    return torch.tensor(rewards, device=pred_traj.device, dtype=torch.float32)


def reward_weighted_loss(
    loss: torch.Tensor,
    rewards: torch.Tensor,
    reward_weight_mode: str = "normalize",
) -> tuple:
    if reward_weight_mode == "normalize":
        if rewards.numel() < 2:
            w = torch.ones_like(rewards)
        else:
            rw_std = rewards.std(unbiased=False)
            w = (rewards - rewards.mean()) / (rw_std + 1e-8)
            w = w - w.min()
            w_max = w.max()
            w = w / (w_max + 1e-8) if w_max > 1e-8 else torch.ones_like(w)
    elif reward_weight_mode == "raw":
        w = rewards.clamp(0.0, 1.0)
    elif reward_weight_mode == "threshold":
        w = (rewards >= rewards.median()).float()
    else:
        raise ValueError(f"Unknown reward_weight_mode: {reward_weight_mode}")

    scalar_w = w.mean()
    scalar_w = torch.nan_to_num(scalar_w, nan=1.0, posinf=1.0, neginf=1.0)
    weighted = loss * scalar_w
    weighted = torch.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)
    return weighted, scalar_w


class ReCogDriveDiTDistillTrainer:
    """
    Reward-weighted DiT distillation with OPD-style chain supervision.

    For DDPM/DDIM:
      - Build student denoising chain (deterministic update for stability)
      - At each step, run teacher on student x_t and match outputs/features

    For flow:
      - Fallback to one-step random-t distillation (current training style)
    """

    def __init__(
        self,
        metric_cache_path: str,
        reward_weight_mode: str = "normalize",
        use_reward_weighting: bool = True,
        lambda_out: float = 1.0,
        lambda_feat: float = 0.2,
        lambda_traj: float = 0.1,
        feat_layers: int = 6,
        out_loss_type: str = "huber",
        scorer_config: Optional[PDMScorerConfig] = None,
    ):
        self.reward_weight_mode = reward_weight_mode
        self.use_reward_weighting = use_reward_weighting
        self.lambda_out = float(lambda_out)
        self.lambda_feat = float(lambda_feat)
        self.lambda_traj = float(lambda_traj)
        self.feat_layers = int(feat_layers)
        self.out_loss_type = out_loss_type

        self.metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        if scorer_config is None:
            scorer_config = PDMScorerConfig(
                progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0
            )
        self.scorer = PDMScorer(proposal_sampling, scorer_config)

    def compute_loss(
        self,
        student_planner,
        teacher_planner,
        vl_features: torch.Tensor,
        action_input: "BatchFeature",
        tokens_list: List[str],
    ) -> BatchFeature:
        if student_planner.config.sampling_method in ["ddpm", "ddim"]:
            out_loss, feat_loss, pred_traj = _chainwise_dit_distill_loss(
                student_planner=student_planner,
                teacher_planner=teacher_planner,
                vl_features=vl_features,
                action_input=action_input,
                out_loss_type=self.out_loss_type,
                feat_layers=self.feat_layers,
            )
        else:
            out_loss, feat_loss, pred_traj = _single_step_dit_distill_loss(
                student_planner=student_planner,
                teacher_planner=teacher_planner,
                vl_features=vl_features,
                action_input=action_input,
                out_loss_type=self.out_loss_type,
                feat_layers=self.feat_layers,
            )

        with torch.no_grad():
            teacher_traj_output = teacher_planner.get_action(vl_features, action_input, deterministic=True)
            teacher_pred_traj = teacher_traj_output["pred_traj"]
            rewards = compute_pdm_rewards(
                pred_traj=pred_traj,
                tokens_list=tokens_list,
                metric_cache_loader=self.metric_cache_loader,
                simulator=self.simulator,
                scorer=self.scorer,
            )

        traj_loss = F.l1_loss(pred_traj.float(), teacher_pred_traj.float())
        distill_loss = (
            self.lambda_out * out_loss
            + self.lambda_feat * feat_loss
            + self.lambda_traj * traj_loss
        )

        if self.use_reward_weighting:
            loss, scalar_w = reward_weighted_loss(distill_loss, rewards, self.reward_weight_mode)
        else:
            loss = distill_loss
            scalar_w = torch.ones(1, device=distill_loss.device)

        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return BatchFeature(data={
            "loss": loss,
            "distill_loss": distill_loss.detach(),
            "out_loss": out_loss.detach(),
            "feat_loss": feat_loss.detach(),
            "traj_loss": traj_loss.detach(),
            "reward_mean": rewards.mean().detach(),
            "reward_weight": scalar_w.detach(),
        })


def _build_common_embeddings(planner, vl_features: torch.Tensor, action_input: "BatchFeature"):
    vl_embeds = planner.feature_encoder(vl_features)
    his_traj_features = planner.his_traj_encoder(
        action_input.his_traj.unsqueeze(1)
    ).repeat(1, planner.config.action_horizon, 1)
    ego_status_features = planner.ego_status_encoder(action_input.status_feature)
    return vl_embeds, his_traj_features, ego_status_features


def _chainwise_dit_distill_loss(
    student_planner,
    teacher_planner,
    vl_features: torch.Tensor,
    action_input: "BatchFeature",
    out_loss_type: str,
    feat_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step-wise DiT distillation on student on-policy DDPM/DDIM denoising chain."""
    vl_embeds_s, his_s, ego_s = _build_common_embeddings(student_planner, vl_features, action_input)
    vl_embeds_t, his_t, ego_t = _build_common_embeddings(teacher_planner, vl_features, action_input)

    B = vl_features.shape[0]
    H = student_planner.config.action_horizon
    D = student_planner.config.action_dim
    device = vl_features.device
    dtype = vl_features.dtype

    current_actions = torch.randn((B, H, D), device=device, dtype=dtype)

    if student_planner.config.sampling_method == "ddpm":
        step_size = student_planner.config.ddpm_cfg.num_train_timesteps // student_planner.config.num_inference_steps
        timesteps = list(reversed(range(0, student_planner.config.ddpm_cfg.num_train_timesteps, step_size)))
    else:
        timesteps = [int(t.item()) for t in student_planner.ddim_t]

    out_losses = []
    feat_losses = []

    for i, t_int in enumerate(timesteps):
        t_batch = student_planner.make_timesteps(B, t_int, device)
        if student_planner.config.sampling_method == "ddim":
            index_batch = student_planner.make_timesteps(B, i, device)
        else:
            index_batch = t_batch

        # teacher/student forward on the same student x_t
        with torch.no_grad():
            teacher_pred, teacher_hiddens = _dit_single_step_forward(
                teacher_planner,
                vl_embeds_t,
                his_t,
                ego_t,
                current_actions,
                t_batch,
                return_hidden_states=True,
            )

        student_pred, student_hiddens = _dit_single_step_forward(
            student_planner,
            vl_embeds_s,
            his_s,
            ego_s,
            current_actions,
            t_batch,
            return_hidden_states=True,
        )

        if out_loss_type == "huber":
            out_l = F.smooth_l1_loss(student_pred.float(), teacher_pred.float())
        elif out_loss_type == "mse":
            out_l = F.mse_loss(student_pred.float(), teacher_pred.float())
        else:
            raise ValueError(f"Unknown out_loss_type: {out_loss_type}")

        feat_l = _feature_distill_loss(student_hiddens, teacher_hiddens, take_last_n=feat_layers)
        out_losses.append(out_l)
        feat_losses.append(feat_l)

        # deterministic reverse update: keep gradients through mean prediction only
        mean, _, _ = student_planner.p_mean_variance(
            current_actions,
            t_batch,
            index_batch,
            vl_embeds_s,
            his_s,
            ego_s,
            deterministic=True,
        )
        current_actions = mean

    final_action_clip_value = getattr(student_planner, "final_action_clip_value", 1.0)
    if final_action_clip_value is not None:
        current_actions = current_actions.clamp(-final_action_clip_value, final_action_clip_value)

    pred_traj = student_planner.denorm_odo(current_actions)
    out_loss = torch.stack(out_losses).mean() if out_losses else torch.tensor(0.0, device=device)
    feat_loss = torch.stack(feat_losses).mean() if feat_losses else torch.tensor(0.0, device=device)
    return out_loss, feat_loss, pred_traj


def _single_step_dit_distill_loss(
    student_planner,
    teacher_planner,
    vl_features: torch.Tensor,
    action_input: "BatchFeature",
    out_loss_type: str,
    feat_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback for flow: random-t one-step distillation."""
    gt_actions = student_planner.norm_odo(action_input.action)
    noise = torch.randn_like(gt_actions)

    t_cont = student_planner.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
    t_cont_reshaped = t_cont[:, None, None]
    noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
    t_discrete = (t_cont * student_planner.num_timestep_buckets).long()

    vl_embeds_s, his_s, ego_s = _build_common_embeddings(student_planner, vl_features, action_input)
    vl_embeds_t, his_t, ego_t = _build_common_embeddings(teacher_planner, vl_features, action_input)

    with torch.no_grad():
        teacher_pred, teacher_hiddens = _dit_single_step_forward(
            teacher_planner, vl_embeds_t, his_t, ego_t, noisy_actions, t_discrete, return_hidden_states=True
        )
    student_pred, student_hiddens = _dit_single_step_forward(
        student_planner, vl_embeds_s, his_s, ego_s, noisy_actions, t_discrete, return_hidden_states=True
    )

    if out_loss_type == "huber":
        out_loss = F.smooth_l1_loss(student_pred.float(), teacher_pred.float())
    elif out_loss_type == "mse":
        out_loss = F.mse_loss(student_pred.float(), teacher_pred.float())
    else:
        raise ValueError(f"Unknown out_loss_type: {out_loss_type}")

    feat_loss = _feature_distill_loss(student_hiddens, teacher_hiddens, take_last_n=feat_layers)
    pred_traj = student_planner.get_action(vl_features, action_input, deterministic=True)["pred_traj"]
    return out_loss, feat_loss, pred_traj


def _dit_single_step_forward(
    planner,
    vl_embeds: torch.Tensor,
    his_traj_features: torch.Tensor,
    ego_status_features: torch.Tensor,
    noisy_actions: torch.Tensor,
    t: torch.Tensor,
    return_hidden_states: bool = False,
) -> Union[Tuple[torch.Tensor, Optional[List[torch.Tensor]]], torch.Tensor]:
    action_features = planner.action_encoder(noisy_actions, t)
    if hasattr(planner, "position_embedding"):
        pos_ids = torch.arange(action_features.shape[1], device=noisy_actions.device)
        action_features = action_features + planner.position_embedding(pos_ids)

    vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, planner.config.action_horizon, 1)
    fused_input = planner.fusion_projector(
        torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
    )

    model_out = planner.model(
        fused_input,
        vl_embeds,
        ego_status_features,
        t,
        return_hidden_states=return_hidden_states,
    )
    if return_hidden_states:
        model_output, all_hidden_states = model_out
    else:
        model_output = model_out
        all_hidden_states = None

    pred = planner.action_decoder(model_output)
    if planner.config.sampling_method == "flow" and planner.config.flow_cfg.mean_variance_net:
        pred = pred.chunk(2, dim=-1)[0]

    if return_hidden_states:
        return pred, all_hidden_states
    return pred


def _feature_distill_loss(
    student_hiddens: List[torch.Tensor],
    teacher_hiddens: List[torch.Tensor],
    take_last_n: int = 6,
) -> torch.Tensor:
    if not student_hiddens or not teacher_hiddens:
        if student_hiddens:
            return torch.zeros((), device=student_hiddens[0].device, dtype=torch.float32)
        if teacher_hiddens:
            return torch.zeros((), device=teacher_hiddens[0].device, dtype=torch.float32)
        return torch.tensor(0.0, dtype=torch.float32)

    n = max(1, min(take_last_n, len(student_hiddens), len(teacher_hiddens)))
    s_list = student_hiddens[-n:]
    t_list = teacher_hiddens[-n:]

    losses = []
    for s_h, t_h in zip(s_list, t_list):
        s_n = F.layer_norm(s_h.float(), s_h.shape[-1:])
        t_n = F.layer_norm(t_h.float(), t_h.shape[-1:])
        losses.append(F.mse_loss(s_n, t_n))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=s_list[0].device)

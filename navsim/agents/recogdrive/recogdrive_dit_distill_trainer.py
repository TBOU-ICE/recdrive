# DiT Distillation trainer for ReCogDrive.
#
# Loss = w(PDM_score) * MSE(student_pred_noise, teacher_pred_noise)
#
# Both teacher and student DiT receive identical inputs:
#   - same vl_embeds (from shared frozen VLM hidden state)
#   - same noisy_actions (sampled at random timestep t)
#   - same t
#
# Teacher DiT is frozen. Only student DiT parameters are updated.
# PDM score of the student's denoised trajectory is used as reward weight.

import logging
import lzma
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

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
    """Return PDM scores as a (B,) float tensor."""
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
            # if all rewards are identical, w_max==0 → treat as uniform weight
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
    Reward-weighted DiT distillation.

    At each training step:
      1. Sample random timestep t and noise.
      2. Compute noisy_actions from GT trajectory.
      3. Teacher DiT (frozen) predicts noise/velocity → target.
      4. Student DiT predicts noise/velocity → pred.
      5. MSE(pred, target) weighted by PDM score of student's full denoised trajectory.
    """

    def __init__(
        self,
        metric_cache_path: str,
        reward_weight_mode: str = "normalize",
        use_reward_weighting: bool = True,
        scorer_config: Optional[PDMScorerConfig] = None,
    ):
        self.reward_weight_mode = reward_weight_mode
        self.use_reward_weighting = use_reward_weighting

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
        student_planner,   # ReCogDriveDiffusionPlanner (trainable)
        teacher_planner,   # ReCogDriveDiffusionPlanner (frozen)
        vl_features: torch.Tensor,       # (B, L, D) shared hidden state
        action_input: "BatchFeature",    # contains his_traj, status_feature, action (GT)
        tokens_list: List[str],
    ) -> BatchFeature:
        """
        Returns BatchFeature with keys: loss, distill_loss, reward_mean, reward_weight
        """
        # ── shared encodings (same for teacher and student) ───────────────────
        model_dtype = vl_features.dtype

        # ── sample noise and timestep ─────────────────────────────────────────
        gt_actions = student_planner.norm_odo(action_input.action)  # (B, H, 3)
        noise = torch.randn_like(gt_actions)

        sampling_method = student_planner.config.sampling_method

        if sampling_method == 'flow':
            t_cont = student_planner.sample_time(
                gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype
            )
            t_cont_reshaped = t_cont[:, None, None]
            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
            velocity_target_gt = gt_actions - noise
            t_discrete = (t_cont * student_planner.num_timestep_buckets).long()
        else:
            t_discrete = student_planner.sample_time(
                gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype
            )
            noisy_actions = (
                student_planner.extract(student_planner.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape) * gt_actions
                + student_planner.extract(student_planner.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape) * noise
            )

        # ── teacher forward (frozen) ──────────────────────────────────────────
        with torch.no_grad():
            teacher_pred = _dit_single_step_forward(teacher_planner, vl_features, action_input, noisy_actions, t_discrete)

        # ── student forward ───────────────────────────────────────────────────
        student_pred = _dit_single_step_forward(student_planner, vl_features, action_input, noisy_actions, t_discrete)

        # ── distillation loss (MSE in noise/velocity space) ───────────────────
        distill_loss = F.mse_loss(student_pred.float(), teacher_pred.float())

        # ── PDM reward from student's full denoised trajectory ────────────────
        with torch.no_grad():
            traj_output = student_planner.get_action(vl_features, action_input)
            pred_traj = traj_output["pred_traj"]  # (B, H, 3)

            rewards = compute_pdm_rewards(
                pred_traj=pred_traj,
                tokens_list=tokens_list,
                metric_cache_loader=self.metric_cache_loader,
                simulator=self.simulator,
                scorer=self.scorer,
            )

        if self.use_reward_weighting:
            loss, scalar_w = reward_weighted_loss(distill_loss, rewards, self.reward_weight_mode)
        else:
            loss = distill_loss
            scalar_w = torch.ones(1, device=distill_loss.device)

        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return BatchFeature(data={
            "loss":          loss,
            "distill_loss":  distill_loss.detach(),
            "reward_mean":   rewards.mean().detach(),
            "reward_weight": scalar_w.detach(),
        })


def _dit_single_step_forward(
    planner,
    vl_features: torch.Tensor,
    action_input: "BatchFeature",
    noisy_actions: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Run one DiT denoising step and return the raw prediction (noise or velocity).
    Mirrors the forward() logic but accepts pre-computed noisy_actions and t.
    """
    vl_embeds = planner.feature_encoder(vl_features)
    his_traj_features = planner.his_traj_encoder(
        action_input.his_traj.unsqueeze(1)
    ).repeat(1, planner.config.action_horizon, 1)
    ego_status_features = planner.ego_status_encoder(action_input.status_feature)

    action_features = planner.action_encoder(noisy_actions, t)
    if hasattr(planner, 'position_embedding'):
        pos_ids = torch.arange(action_features.shape[1], device=noisy_actions.device)
        action_features = action_features + planner.position_embedding(pos_ids)

    vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, planner.config.action_horizon, 1)
    fused_input = planner.fusion_projector(
        torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
    )

    model_output = planner.model(fused_input, vl_embeds, ego_status_features, t)
    pred = planner.action_decoder(model_output)

    # for flow matching, only take the velocity head (first half)
    if planner.config.sampling_method == 'flow' and planner.config.flow_cfg.mean_variance_net:
        pred = pred.chunk(2, dim=-1)[0]

    return pred

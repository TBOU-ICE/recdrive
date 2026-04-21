# OPD (On-Policy Distillation) trainer for ReCogDrive VLA.
#
# Loss = w(PDM_score) * L_vlm_opd
#
# L_vlm_opd: Teacher-TopK truncated reverse-KL between 8B teacher VLM and 2B student VLM,
#             computed on the student's own on-policy generated token sequences.
# w(PDM_score): reward-weighted coefficient — samples with higher PDM scores
#               contribute more to the distillation loss.

import lzma
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

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


# ──────────────────────────────────────────────────────────────────────────────
# Teacher-TopK truncated reverse-KL
# ──────────────────────────────────────────────────────────────────────────────

def compute_topk_kl_loss(
    student_logits: torch.Tensor,   # (B, T, V)
    teacher_logits: torch.Tensor,   # (B, T, V)
    response_mask: torch.Tensor,    # (B, T)  1 = valid response token
    topk: int = 32,
    norm_to_one: bool = True,
) -> torch.Tensor:
    """
    Teacher-TopK truncated reverse-KL: KL(student_norm || teacher_norm)
    restricted to the teacher's top-K token support at each position.
    Returns scalar loss averaged over valid tokens.
    """
    # teacher top-K indices
    _, ref_topk_indices = teacher_logits.topk(topk, dim=-1)           # (B, T, K)

    # gather logits at teacher top-K positions
    ref_logits_k   = teacher_logits.gather(-1, ref_topk_indices)       # (B, T, K)
    actor_logits_k = student_logits.gather(-1, ref_topk_indices)       # (B, T, K)

    if norm_to_one:
        teacher_log_norm  = F.log_softmax(ref_logits_k,   dim=-1)
        student_log_norm  = F.log_softmax(actor_logits_k, dim=-1)
        student_prob_norm = student_log_norm.exp()
        kl_per_token = (student_prob_norm * (student_log_norm - teacher_log_norm)).sum(dim=-1)
    else:
        ref_logsumexp   = teacher_logits.logsumexp(dim=-1, keepdim=True)
        actor_logsumexp = student_logits.logsumexp(dim=-1, keepdim=True)
        log_p_k = actor_logits_k - actor_logsumexp
        log_q_k = ref_logits_k   - ref_logsumexp
        p_k = log_p_k.exp()
        kl_per_token = (p_k * (log_p_k - log_q_k)).sum(dim=-1)

    kl_per_token = kl_per_token * response_mask
    n_valid = response_mask.sum().clamp(min=1)
    return kl_per_token.sum() / n_valid


# ──────────────────────────────────────────────────────────────────────────────
# PDM reward computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_pdm_rewards(
    pred_traj: torch.Tensor,
    tokens_list: List[str],
    metric_cache_loader: MetricCacheLoader,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> torch.Tensor:
    """Return PDM scores as a (B,) float tensor."""
    pred_np = pred_traj.detach().cpu().numpy()
    unique_tokens = set(tokens_list)
    cache_dict = {}
    for token in unique_tokens:
        path = metric_cache_loader.metric_cache_paths[token]
        with lzma.open(path, "rb") as f:
            cache_dict[token] = pickle.load(f)

    rewards = []
    for i, token in enumerate(tokens_list):
        traj = Trajectory(pred_np[i])
        result = pdm_score(
            metric_cache=cache_dict[token],
            model_trajectory=traj,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        rewards.append(asdict(result)["score"])

    return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Reward weighting
# ──────────────────────────────────────────────────────────────────────────────

def reward_weighted_opd_loss(
    kl_loss: torch.Tensor,
    rewards: torch.Tensor,          # (B,)
    reward_weight_mode: str = "normalize",
) -> tuple:
    """
    Scale the OPD loss by a reward-derived weight.

    Modes:
      "normalize" : shift-normalize rewards to [0,1], use mean as scalar weight
      "raw"       : use PDM score directly (already in [0,1])
      "threshold" : 1 if score > median else 0
    """
    if reward_weight_mode == "normalize":
        w = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        w = w - w.min()
        w = w / (w.max() + 1e-8)
    elif reward_weight_mode == "raw":
        w = rewards.clamp(0.0, 1.0)
    elif reward_weight_mode == "threshold":
        w = (rewards >= rewards.median()).float()
    else:
        raise ValueError(f"Unknown reward_weight_mode: {reward_weight_mode}")

    scalar_w = w.mean()
    return kl_loss * scalar_w, scalar_w


# ──────────────────────────────────────────────────────────────────────────────
# Main OPD trainer
# ──────────────────────────────────────────────────────────────────────────────

class ReCogDriveOPDTrainer:
    """
    Encapsulates reward-weighted OPD training logic.

    Called from ReCogDriveAgent.forward_opd().
    """

    def __init__(
        self,
        metric_cache_path: str,
        topk: int = 32,
        norm_to_one: bool = True,
        reward_weight_mode: str = "normalize",
        use_reward_weighting: bool = True,
        scorer_config: Optional[PDMScorerConfig] = None,
    ):
        self.topk = topk
        self.norm_to_one = norm_to_one
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
        student_logits: torch.Tensor,   # (B, T, V)
        teacher_logits: torch.Tensor,   # (B, T, V)
        response_mask: torch.Tensor,    # (B, T)
        pred_traj: torch.Tensor,        # (B, H, 3) denormalized
        tokens_list: List[str],
    ) -> BatchFeature:
        """
        Returns BatchFeature with keys: loss, opd_loss, reward_mean, reward_weight
        """
        opd_loss = compute_topk_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            response_mask=response_mask,
            topk=self.topk,
            norm_to_one=self.norm_to_one,
        )

        with torch.no_grad():
            rewards = compute_pdm_rewards(
                pred_traj=pred_traj,
                tokens_list=tokens_list,
                metric_cache_loader=self.metric_cache_loader,
                simulator=self.simulator,
                scorer=self.scorer,
            )

        if self.use_reward_weighting:
            loss, scalar_w = reward_weighted_opd_loss(opd_loss, rewards, self.reward_weight_mode)
        else:
            loss = opd_loss
            scalar_w = torch.ones(1, device=opd_loss.device)

        return BatchFeature(data={
            "loss":          loss,
            "opd_loss":      opd_loss.detach(),
            "reward_mean":   rewards.mean().detach(),
            "reward_weight": scalar_w.detach() if isinstance(scalar_w, torch.Tensor) else scalar_w,
        })

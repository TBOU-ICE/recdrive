"""
Expert-Reward RL Agent: expert IL anchor (strong supervised signal).

Problem context
---------------
After BC + GRPO-PDMS training, PDMS improves but EPDMS and EC drop because
the PDMS reward does not penalise deviation from natural, expert-quality
trajectories.  The pseudo-expert dataset provides, for each scene token, a
set of trajectories pre-scored by the closed-loop PDMS simulator; the best
one (PDMS=1.0 for ~72% of tokens) is simultaneously safe *and* physically
natural.

Reward
    combined_reward = pdms_weight  * PDMS
                    + temporal_weight * temporal_consistency   (inherited)

Expert IL anchor
    expert_il_loss = planner.forward(vl, expert_action_input)["loss"]
    i.e. the DiT's standard DDPM/flow denoising loss with the best-PDMS
    pseudo-expert trajectory as x_0.  A high weight (0.5) makes this the
    primary signal that prevents EC degradation, replacing both old-policy BC
    and the earlier EC reward term.

Total loss
    L = GRPO_policy_loss(combined_reward)
      + expert_il_weight * expert_il_loss

Tokens that are absent from the pseudo-expert lookup (no valid scores) are
handled gracefully: expert IL loss skips those items.
"""

import lzma
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.feature_extraction_utils import BatchFeature

from .recogdrive_temporal_reward_rl_agent import ReCogDriveTemporalRewardRLAgent
from .pseudo_expert_loader import build_expert_lookup

logger = logging.getLogger(__name__)


class ReCogDriveExpertRewardRLAgent(ReCogDriveTemporalRewardRLAgent):
    """
    GRPO RL agent combining EC-aware reward (Plan A) and expert IL anchor (Plan B).

    Inherits the temporal-pair batch format [cur_0, next_0, cur_1, next_1, …]
    and the temporal consistency reward from ReCogDriveTemporalRewardRLAgent.
    Adds EC reward and expert IL loss on top.
    """

    def __init__(
        self,
        *args: Any,
        # Path to the pseudo-expert pkl (dataset_decoupled_v2_clean.pkl)
        expert_data_path: str = "",
        # Weight on the expert denoising IL loss (primary EC-preservation signal)
        expert_il_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._expert_data_path  = expert_data_path
        self._expert_il_weight  = float(expert_il_weight)
        self._expert_lookup: Optional[Dict[str, Any]] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        super().initialize()
        if self._expert_data_path:
            self._expert_lookup = build_expert_lookup(self._expert_data_path)
        else:
            logger.warning(
                "ReCogDriveExpertRewardRLAgent: expert_data_path is empty. "
                "EC reward and expert IL loss will be disabled (zero)."
            )
            self._expert_lookup = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: fetch expert trajectories for a batch of tokens
    # ─────────────────────────────────────────────────────────────────────────

    def _get_expert_trajs(
        self,
        tokens: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            expert_trajs : (B, 8, 3)  ego-relative [x(m), y(m), heading(rad)]
            valid_mask   : (B,)  bool, False for tokens absent from the lookup
        """
        B = len(tokens)
        trajs = torch.zeros(B, 8, 3, device=device, dtype=dtype)
        valid = torch.zeros(B, dtype=torch.bool, device=device)
        for i, tok in enumerate(tokens):
            if tok in self._expert_lookup:
                trajs[i] = torch.tensor(
                    self._expert_lookup[tok], device=device, dtype=dtype
                )
                valid[i] = True
        return trajs, valid

    # ─────────────────────────────────────────────────────────────────────────
    # Expert IL denoising loss
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_expert_il_loss(
        self,
        planner,
        vl_features: torch.Tensor,    # (B, N, C)  raw last_hidden_state
        his_traj: torch.Tensor,       # (B, T*3)   flat history
        ego_status: torch.Tensor,     # (B, 8)
        expert_trajs: torch.Tensor,   # (B, 8, 3)  raw ego-relative
        valid_mask: torch.Tensor,     # (B,)  bool
    ) -> torch.Tensor:
        """
        Standard DDPM/flow denoising loss (planner.forward) using the
        pseudo-expert trajectory as x_0.  Only valid tokens contribute.
        Returns a scalar; returns 0 if no valid tokens in the batch.
        """
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            return vl_features.new_zeros(())

        # Subset to valid tokens to avoid polluting the loss with zero trajs
        vl_v    = vl_features[valid_mask]
        his_v   = his_traj[valid_mask]
        sta_v   = ego_status[valid_mask]
        traj_v  = expert_trajs[valid_mask]   # (n_valid, 8, 3)  raw metres

        expert_action_input = BatchFeature(data={
            "his_traj":       his_v,
            "status_feature": sta_v,
            # planner.forward internally calls norm_odo on this field
            "action":         traj_v,
        })
        result = planner.forward(vl_v, expert_action_input)
        return result["loss"]

    # ─────────────────────────────────────────────────────────────────────────
    # Main forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets=None,
        tokens_list=None,
    ):
        # Inference / validation: delegate entirely to base class (no change).
        if not (self.training and self.grpo):
            return super().forward(features, targets, tokens_list)

        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype        = next(self.action_head.parameters()).dtype
        history_trajectory = features["history_trajectory"].cuda()
        status_feature     = features["status_feature"].cuda()
        last_hidden_state  = features["last_hidden_state"].cuda()

        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)

        his_flat = history_trajectory.view(history_trajectory.size(0), -1)

        action_inputs = BatchFeature(data={
            "state":          torch.cat([status_feature, his_flat], dim=1).to(model_dtype),
            "his_traj":       his_flat.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
            "action":         targets["trajectory"].to(model_dtype),
        })

        planner = self.action_head
        planner.set_frozen_modules_to_eval_mode()

        B      = last_hidden_state.shape[0]
        G      = self._grpo_sample_time
        device = last_hidden_state.device

        vl  = last_hidden_state.to(model_dtype)
        his = his_flat.to(model_dtype)
        sta = status_feature.to(model_dtype)

        # ── Pseudo-expert trajectories for this batch ─────────────────────────
        expert_trajs, expert_valid = self._get_expert_trajs(
            tokens_list, device, model_dtype
        )   # (B, 8, 3),  (B,)

        # ── Expand B → B*G  (repeat_interleave preserves pair layout) ─────────
        vl_rep  = vl.repeat_interleave(G, 0)
        his_rep = his.repeat_interleave(G, 0)
        sta_rep = sta.repeat_interleave(G, 0)

        # ── Sample G rollouts per token ───────────────────────────────────────
        chains, trajs = planner.sample_chain(
            vl_rep, his_rep, sta_rep, deterministic=False
        )   # chains: (B*G, K+1, H, D)  trajs: (B*G, H, 3) denorm, detached

        # ── PDMS reward ───────────────────────────────────────────────────────
        tokens_rep   = [tok for tok in tokens_list for _ in range(G)]
        unique_tokens = set(tokens_list)
        metric_cache  = {}
        for token in unique_tokens:
            path = planner.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, "rb") as f:
                metric_cache[token] = pickle.load(f)
        pdms_rewards = planner.reward_fn(trajs, tokens_rep, metric_cache)   # (B*G,)

        # ── Temporal consistency reward  (inherited from parent class) ────────
        temporal_rewards = self._compute_temporal_reward(trajs, his_rep, B, G)
        if self._normalize_temporal and temporal_rewards.numel() > 1:
            t_min = temporal_rewards.min()
            t_max = temporal_rewards.max()
            temporal_norm = (temporal_rewards - t_min) / (t_max - t_min).clamp(min=1e-8)
        else:
            temporal_norm = temporal_rewards

        # ── Combined reward ───────────────────────────────────────────────────
        combined = (
            self._pdms_weight              * pdms_rewards
            + self._temporal_reward_weight * temporal_norm
        )

        # ── GRPO advantage (per-token normalisation) ──────────────────────────
        rewards_matrix = combined.view(B, G)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r  = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_matrix - mean_r) / std_r).view(-1).detach()

        adv_min = torch.quantile(advantages, planner.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, planner.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        num_denoising_steps = chains.shape[1] - 1
        denoising_idx = torch.arange(num_denoising_steps, device=advantages.device)
        discount      = planner.gamma_denoising ** (num_denoising_steps - denoising_idx - 1)

        adv_steps  = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)
        disc_steps = discount.view(1, 1, -1).expand(B, G, num_denoising_steps)
        adv_weighted = (adv_steps * disc_steps).reshape(-1)

        # ── Policy loss ───────────────────────────────────────────────────────
        log_probs = planner.get_logprobs(
            vl_rep, his_rep, sta_rep, chains, deterministic=False
        )
        log_probs   = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        policy_loss = -torch.mean(log_probs * adv_weighted)
        total_loss  = policy_loss

        # ── Plan B: expert IL loss  (replaces old-policy BC) ─────────────────
        expert_il_loss = self._compute_expert_il_loss(
            planner, vl, his, sta, expert_trajs, expert_valid
        )
        total_loss = total_loss + self._expert_il_weight * expert_il_loss

        return BatchFeature(data={
            "loss":                      total_loss,
            "reward":                    pdms_rewards.mean(),
            "policy_loss":               policy_loss,
            "expert_il_loss":            expert_il_loss.detach(),
            "temporal_reward_mean":      temporal_rewards.mean(),
            "temporal_reward_norm_mean": temporal_norm.mean(),
            "combined_reward_mean":      combined.mean(),
            "expert_valid_fraction":     expert_valid.float().mean(),
        })

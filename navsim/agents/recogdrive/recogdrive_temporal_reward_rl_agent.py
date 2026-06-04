"""
Plan B – GRPO RL training with temporal consistency as part of the REWARD.

    combined_reward = pdms_weight · PDMS_score
                    + temporal_weight · temporal_consistency_score

For every pair (cur_p, next_p) in the temporal pair batch and every rollout g:
  - PDMS reward is computed per token as in the base ReCogDriveAgent.
  - temporal_consistency_score(p, g) measures how well traj_cur_p_g (after
    coordinate transformation into next's ego frame) overlaps with
    traj_next_p_g.  Higher = more consistent.
  - Both tokens in the pair receive the same temporal reward component so
    the policy is jointly incentivised for consistency.

Gradient flow:
  All gradients flow only through the GRPO policy gradient (log_probs).
  Rewards (PDMS + temporal) are always detached.  This is purely RL – no
  supervised signal on trajectories.

Batch format: [cur_0, next_0, cur_1, next_1, …] as produced by
TemporalCachePairDataset + temporal_pair_collate_fn.
"""

import lzma
import pickle
from typing import Any, Dict

import torch
from transformers.feature_extraction_utils import BatchFeature

from .recogdrive_agent import ReCogDriveAgent


class ReCogDriveTemporalRewardRLAgent(ReCogDriveAgent):
    """GRPO RL agent with temporal consistency as reward component (Plan B).

    Re-implements the GRPO forward pass at the agent level so the reward
    signal can be extended with temporal consistency without touching the
    existing ReCogDriveDiffusionPlanner.
    """

    def __init__(
        self,
        *args: Any,
        rl_pdms_weight: float = 1.0,
        rl_temporal_reward_weight: float = 0.1,
        rl_temporal_shift_steps: int = 1,
        rl_temporal_pos_weight: float = 1.0,
        rl_temporal_heading_weight: float = 0.2,
        rl_temporal_normalize_temporal: bool = True,
        rl_grpo_sample_time: int = 8,
        rl_bc_coeff: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.grpo:
            raise ValueError("ReCogDriveTemporalRewardRLAgent requires grpo=True.")

        self._pdms_weight = float(rl_pdms_weight)
        self._temporal_reward_weight = float(rl_temporal_reward_weight)
        self._temporal_shift_steps = int(rl_temporal_shift_steps)
        self._temporal_pos_weight = float(rl_temporal_pos_weight)
        self._temporal_heading_weight = float(rl_temporal_heading_weight)
        self._normalize_temporal = bool(rl_temporal_normalize_temporal)
        self._grpo_sample_time = int(rl_grpo_sample_time)
        self._bc_coeff = float(rl_bc_coeff)

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry helpers (mirrored from ReCogDriveTemporalDiTDistillTrainer)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _transform_to_next_frame(
        traj_cur: torch.Tensor,         # (G, H, 3)
        cur_pose_in_next: torch.Tensor, # (G, 3)  pose of cur ego in next ego frame
    ) -> torch.Tensor:
        """Rigid-body transform: cur ego frame → next ego frame."""
        x, y, h = traj_cur[..., 0], traj_cur[..., 1], traj_cur[..., 2]
        dx = cur_pose_in_next[:, 0:1]   # (G, 1)  broadcast over H
        dy = cur_pose_in_next[:, 1:2]
        dh = cur_pose_in_next[:, 2:3]
        cos_h, sin_h = torch.cos(dh), torch.sin(dh)
        x_n = cos_h * x - sin_h * y + dx
        y_n = sin_h * x + cos_h * y + dy
        h_n = ReCogDriveTemporalRewardRLAgent._angle_wrap(h + dh)
        return torch.stack([x_n, y_n, h_n], dim=-1)   # (G, H, 3)

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal reward computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_temporal_reward(
        self,
        trajs: torch.Tensor,        # (B*G, H, 3) denorm trajectories, detached
        his_traj_rep: torch.Tensor, # (B*G, 12)   flat history, raw
        B: int,
        G: int,
    ) -> torch.Tensor:
        """Temporal consistency reward, one scalar per token×rollout.

        Pair layout in the B-dimension: [cur0, next0, cur1, next1, ...]
        After repeat_interleave(G):
          cur_p  rollout g  →  index (2p)   * G + g
          next_p rollout g  →  index (2p+1) * G + g

        Both tokens in a pair receive the same temporal reward for rollout g
        (the reward is a property of the pair, not individual tokens).

        Returns: (B*G,) temporal reward (higher = more consistent), detached.
        """
        device = trajs.device
        temporal_rew = torch.zeros(B * G, device=device, dtype=trajs.dtype)
        num_pairs = B // 2
        if num_pairs == 0:
            return temporal_rew

        H = trajs.shape[1]
        shift = max(1, min(self._temporal_shift_steps, H - 1))
        hist_3d = his_traj_rep.reshape(B * G, -1, 3)  # (B*G, T_hist, 3)
        T_hist = hist_3d.shape[1]
        hist_ref_idx = -2 if T_hist >= 2 else -1

        for p in range(num_pairs):
            c0 = (2 * p) * G
            n0 = (2 * p + 1) * G

            traj_c = trajs[c0: c0 + G]          # (G, H, 3)
            traj_n = trajs[n0: n0 + G]          # (G, H, 3)
            hist_n = hist_3d[n0: n0 + G]        # (G, T_hist, 3)

            # next-token history[-2] = cur ego pose expressed in next ego frame
            cur_pose = hist_n[:, hist_ref_idx, :]           # (G, 3)
            cur_in_next = self._transform_to_next_frame(traj_c, cur_pose)  # (G, H, 3)

            # Overlapping horizon after time-shift
            c_ov = cur_in_next[:, shift:]                   # (G, H-shift, 3)
            n_ov = traj_n[:, : c_ov.shape[1]]              # (G, H-shift, 3)
            min_len = min(c_ov.shape[1], n_ov.shape[1])
            if min_len <= 0:
                continue
            c_ov = c_ov[:, :min_len]
            n_ov = n_ov[:, :min_len]

            # Per-rollout position and heading error  (G,)
            pos_err  = (c_ov[..., :2] - n_ov[..., :2]).abs().mean(dim=[1, 2])
            head_err = self._angle_wrap(c_ov[..., 2] - n_ov[..., 2]).abs().mean(dim=1)

            inconsistency = (
                self._temporal_pos_weight * pos_err
                + self._temporal_heading_weight * head_err
            )                                               # (G,)

            # Reward = negative inconsistency (higher = better)
            rew_g = -inconsistency
            temporal_rew[c0: c0 + G] = rew_g
            temporal_rew[n0: n0 + G] = rew_g

        return temporal_rew.detach()

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets=None,
        tokens_list=None,
    ):
        # Inference / validation: delegate entirely to base class.
        if not (self.training and self.grpo):
            return super().forward(features, targets, tokens_list)

        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype
        history_trajectory = features["history_trajectory"].cuda()
        status_feature = features["status_feature"].cuda()
        last_hidden_state = features["last_hidden_state"].cuda()

        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)

        his_flat = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, his_flat], dim=1)

        action_inputs = BatchFeature(data={
            "state": input_state.to(model_dtype),
            "his_traj": his_flat.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
            "action": targets["trajectory"].to(model_dtype),
        })

        planner = self.action_head
        planner.set_frozen_modules_to_eval_mode()

        B = last_hidden_state.shape[0]
        G = self._grpo_sample_time

        vl = last_hidden_state.to(model_dtype)
        his = his_flat.to(model_dtype)
        sta = status_feature.to(model_dtype)

        # Expand B → B*G by repeating each sample G times (repeat_interleave
        # keeps pair order: [cur0]*G, [next0]*G, [cur1]*G, [next1]*G, ...)
        vl_rep  = vl.repeat_interleave(G, 0)
        his_rep = his.repeat_interleave(G, 0)
        sta_rep = sta.repeat_interleave(G, 0)

        # ── Sample G rollouts per token ───────────────────────────────────────
        chains, trajs = planner.sample_chain(
            vl_rep, his_rep, sta_rep, deterministic=False
        )   # chains: (B*G, K+1, H, D) detached; trajs: (B*G, H, 3) detached

        # ── PDMS reward ───────────────────────────────────────────────────────
        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        unique_tokens = set(tokens_list)
        metric_cache = {}
        for token in unique_tokens:
            path = planner.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, "rb") as f:
                metric_cache[token] = pickle.load(f)
        pdms_rewards = planner.reward_fn(trajs, tokens_rep, metric_cache)  # (B*G,)

        # ── Temporal consistency reward ───────────────────────────────────────
        temporal_rewards = self._compute_temporal_reward(trajs, his_rep, B, G)  # (B*G,)

        # Min-max normalise temporal reward to [0, 1] so its scale matches PDMS.
        if self._normalize_temporal and temporal_rewards.numel() > 1:
            t_min = temporal_rewards.min()
            t_max = temporal_rewards.max()
            t_range = (t_max - t_min).clamp(min=1e-8)
            temporal_norm = (temporal_rewards - t_min) / t_range
        else:
            temporal_norm = temporal_rewards

        # ── Combined reward ───────────────────────────────────────────────────
        combined = (
            self._pdms_weight * pdms_rewards
            + self._temporal_reward_weight * temporal_norm
        )

        # ── GRPO advantage (per-token normalisation, identical to forward_grpo)
        rewards_matrix = combined.view(B, G)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r  = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_matrix - mean_r) / std_r).view(-1).detach()

        adv_min = torch.quantile(advantages, planner.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, planner.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        num_denoising_steps = chains.shape[1] - 1
        denoising_idx = torch.arange(num_denoising_steps, device=advantages.device)
        discount = planner.gamma_denoising ** (num_denoising_steps - denoising_idx - 1)

        adv_steps       = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)
        discount_expand = discount.view(1, 1, -1).expand(B, G, num_denoising_steps)
        adv_weighted    = (adv_steps * discount_expand).reshape(-1)

        # ── Policy loss ───────────────────────────────────────────────────────
        log_probs = planner.get_logprobs(
            vl_rep, his_rep, sta_rep, chains, deterministic=False
        )
        log_probs   = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        policy_loss = -torch.mean(log_probs * adv_weighted)
        total_loss  = policy_loss

        # ── BC loss (identical to forward_grpo) ───────────────────────────────
        with torch.no_grad():
            teacher_chains, _ = planner.old_policy.sample_chain(
                vl, his, sta, deterministic=False
            )
        bc_logp = planner.get_logprobs(vl, his, sta, teacher_chains, deterministic=False)
        bc_logp = bc_logp.clamp(min=-5, max=2)
        K_steps = chains.shape[1] - 1
        bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(
            dim=[1, 2, 3]
        )
        bc_loss    = -bc_logp.mean()
        total_loss = total_loss + self._bc_coeff * bc_loss

        return BatchFeature(data={
            "loss":                    total_loss,
            "reward":                  pdms_rewards.mean(),
            "policy_loss":             policy_loss,
            "bc_loss":                 bc_loss,
            "temporal_reward_mean":    temporal_rewards.mean(),
            "temporal_reward_norm_mean": temporal_norm.mean(),
            "combined_reward_mean":    combined.mean(),
        })

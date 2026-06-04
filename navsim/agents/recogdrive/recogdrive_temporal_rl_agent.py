"""
Plan A – GRPO RL training with temporal consistency as an AUXILIARY LOSS.

    total_loss = GRPO_policy_loss(PDMS_reward) + λ · temporal_consistency_loss

The GRPO policy loss is computed exactly as in the base ReCogDriveAgent
(forward_grpo on the action_head).  After that, one additional grad-enabled
DDIM forward pass produces the predicted trajectory; the temporal loss is
computed on it and added to the GRPO loss.

Batch format: [cur_0, next_0, cur_1, next_1, …] as produced by
TemporalCachePairDataset + temporal_pair_collate_fn.

Gradient flow:
  - GRPO gradient: PDMS reward → advantage → policy_loss → action_head
  - Temporal gradient: last-step DDIM mu → denorm → temporal_loss → action_head
  Both gradients update the same action_head parameters in one backward pass.
"""

from typing import Any, Dict

import torch
from transformers.feature_extraction_utils import BatchFeature

from .recogdrive_agent import ReCogDriveAgent
from .recogdrive_dit_temporal_distill_trainer import ReCogDriveTemporalDiTDistillTrainer


class ReCogDriveTemporalRLAgent(ReCogDriveAgent):
    """GRPO RL agent + temporal consistency auxiliary loss (Plan A).

    Drop-in replacement for ReCogDriveAgent when grpo=True and training on
    temporal pair data.  All inference / eval behaviour is unchanged.
    """

    def __init__(
        self,
        *args: Any,
        rl_temporal_loss_weight: float = 0.05,
        rl_temporal_shift_steps: int = 1,
        rl_temporal_pos_weight: float = 1.0,
        rl_temporal_heading_weight: float = 0.2,
        rl_temporal_acc_weight: float = 0.1,
        rl_temporal_jerk_weight: float = 0.05,
        rl_temporal_yaw_rate_weight: float = 0.1,
        rl_temporal_yaw_acc_weight: float = 0.05,
        rl_temporal_dt: float = 0.5,
        rl_temporal_min_sigma: float = 0.04,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.grpo:
            raise ValueError("ReCogDriveTemporalRLAgent requires grpo=True.")

        # Reuse the temporal loss computation from the distillation trainer.
        # We only call _compute_temporal_loss(); the teacher-dependent
        # compute_loss() is never invoked so no teacher is needed.
        self._temporal_helper = ReCogDriveTemporalDiTDistillTrainer(
            temporal_loss_weight=rl_temporal_loss_weight,
            temporal_shift_steps=rl_temporal_shift_steps,
            temporal_pos_weight=rl_temporal_pos_weight,
            temporal_heading_weight=rl_temporal_heading_weight,
            temporal_acc_weight=rl_temporal_acc_weight,
            temporal_jerk_weight=rl_temporal_jerk_weight,
            temporal_yaw_rate_weight=rl_temporal_yaw_rate_weight,
            temporal_yaw_acc_weight=rl_temporal_yaw_acc_weight,
            temporal_dt=rl_temporal_dt,
        )
        self._rl_temporal_min_sigma = float(rl_temporal_min_sigma)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _student_traj_with_grad(
        self,
        vl_features: torch.Tensor,     # (B, N, d)  raw last_hidden_state
        his_traj: torch.Tensor,        # (B, 12)    flat history
        status_feature: torch.Tensor,  # (B, 8)
    ) -> torch.Tensor:
        """Run DDIM chain and return the predicted trajectory WITH gradient.

        Runs the first (K-1) denoising steps under no_grad for efficiency,
        keeps the intermediate state, then runs the final step with gradients
        to produce a differentiable trajectory for the temporal loss.

        Returns: (B, H, 3) denormalized ego-frame trajectory, grad-enabled.
        """
        planner = self.action_head
        dtype = next(planner.parameters()).dtype
        B = vl_features.shape[0]
        device = vl_features.device
        K = planner.ddim_steps
        H, D = planner.config.action_horizon, planner.config.action_dim

        # ── (K-1) steps: explore without storing grad ─────────────────────────
        with torch.no_grad():
            vl_e, his_e, ego_e = ReCogDriveTemporalDiTDistillTrainer._encode(
                planner, vl_features, his_traj, status_feature, dtype
            )
            z = torch.randn((B, H, D), device=device, dtype=dtype)
            min_s = self._rl_temporal_min_sigma
            for i in range(K - 1):
                t_b = planner.make_timesteps(B, int(planner.ddim_t[i].item()), device)
                i_b = planner.make_timesteps(B, i, device)
                mu, logvar, _ = planner.p_mean_variance(
                    z, t_b, i_b, vl_e, his_e, ego_e, deterministic=False
                )
                sigma = (
                    (0.5 * logvar.float().clamp(-20.0, 20.0))
                    .exp().clamp(min=min_s).to(dtype)
                )
                z = (mu + sigma * torch.randn_like(z).clamp_(-5.0, 5.0)).detach()

        # ── Final step: grad-enabled forward ──────────────────────────────────
        vl_eg, his_eg, ego_eg = ReCogDriveTemporalDiTDistillTrainer._encode(
            planner, vl_features, his_traj, status_feature, dtype
        )
        t_last = planner.make_timesteps(B, int(planner.ddim_t[K - 1].item()), device)
        i_last = planner.make_timesteps(B, K - 1, device)
        mu_last, _, _ = planner.p_mean_variance(
            z, t_last, i_last, vl_eg, his_eg, ego_eg, deterministic=True
        )
        # denorm_odo is linear → grad flows through unchanged
        return planner.denorm_odo(mu_last.float())  # (B, H, 3)

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

        # Move to GPU (same as base class).
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

        vl = last_hidden_state.to(model_dtype)

        # ── GRPO policy loss (identical to base class) ────────────────────────
        grpo_out = self.action_head.forward_grpo(vl, action_inputs, tokens_list)

        # ── Temporal auxiliary loss (one extra grad-enabled DDIM forward) ─────
        pred_traj = self._student_traj_with_grad(
            vl, his_flat.to(model_dtype), status_feature.to(model_dtype)
        )
        temporal_out = self._temporal_helper._compute_temporal_loss(
            pred_traj, his_flat.float()
        )

        total_loss = grpo_out.loss + temporal_out["temporal_loss"]
        if not torch.isfinite(total_loss):
            total_loss = grpo_out.loss  # fall back to pure GRPO if NaN

        return BatchFeature(data={
            "loss": total_loss,
            "reward": grpo_out["reward"],
            "policy_loss": grpo_out["policy_loss"],
            "bc_loss": grpo_out["bc_loss"],
            **temporal_out,
        })

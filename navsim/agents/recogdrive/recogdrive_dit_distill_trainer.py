"""
DiT OPD (On-Policy Distillation) trainer for ReCogDrive.

Adapted from Flow-OPD (arXiv:2605.08063) to the DDIM denoising setting.

The denoising chain z_T → z_{T-1} → ... → z_0 is the "policy trajectory."
Per-step KL between student and teacher denoising distributions serves as the
advantage signal for PPO-clip policy gradient — no outcome reward needed.

Algorithm:
  1. Sample student on-policy chain via stochastic DDIM (no_grad).
  2. Compute log_prob_old under student policy for each chain step (no_grad).
  3. For each denoising step i:
       a. Student forward (grad): μ_θ(z_t, t, h)
       b. Teacher forward (frozen, deterministic): μ_φ(z_t, t, h)
       c. KL advantage:  adv_i = -‖μ_θ − μ_φ‖² / (2σ_θ²)
       d. Normalise advantage over the batch.
       e. PPO-clip:  loss_i = max(−adv · r,  −adv · clip(r, 1−ε, 1+ε))
                     where  r = exp(log_prob_new − log_prob_old)
  4. Total loss = mean over K steps.

Teacher and student both receive the same raw vl_features (B, N, 1536) from
the cached VLM hidden states, each encoding it through their own feature_encoder.
"""

import torch
from torch.distributions import Normal
from transformers.feature_extraction_utils import BatchFeature


class ReCogDriveDiTOPDTrainer:
    """
    Flow-OPD-style on-policy distillation between two DiT planners.

    No PDM reward, no feature distillation — pure denoising-transition KL.
    """

    def __init__(
        self,
        eps_clip: float = 0.2,
        min_sigma: float = 0.04,
        min_logprob_sigma: float = 0.1,
        normalize_advantage: bool = True,
    ):
        self.eps_clip = eps_clip
        self.min_sigma = min_sigma
        self.min_logprob_sigma = min_logprob_sigma
        self.normalize_advantage = normalize_advantage

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _encode(planner, vl_features, his_traj, ego_status, dtype):
        """
        Encode raw conditioning tensors through the planner's learned projections.

        Args:
            planner:     ReCogDriveDiffusionPlanner
            vl_features: (B, N, 1536)  raw VLM hidden states
            his_traj:    (B, 12)       flattened history trajectory
            ego_status:  (B, 8)        ego status features
            dtype:       target dtype for the planner

        Returns:
            vl_embeds  (B, N, D)
            his_embeds (B, H, D)
            ego_embeds (B, D)
        """
        vl_embeds  = planner.feature_encoder(vl_features.to(dtype))
        his_embeds = (
            planner.his_traj_encoder(his_traj.to(dtype).unsqueeze(1))
            .repeat(1, planner.config.action_horizon, 1)
        )
        ego_embeds = planner.ego_status_encoder(ego_status.to(dtype))
        return vl_embeds, his_embeds, ego_embeds

    def _sample_chain(self, planner, vl_embeds, his_embeds, ego_embeds, B, device, dtype):
        """
        Generate an on-policy stochastic DDIM denoising chain.

        Uses EtaFixed (≈1.0) for exploration — analogous to SDE sampling in
        Flow-OPD.  All tensors are detached; no gradient is tracked.

        Returns:
            chain (B, K+1, H, D)  — all detached
        """
        H, D = planner.config.action_horizon, planner.config.action_dim
        z = torch.randn((B, H, D), device=device, dtype=dtype)
        chain = [z.clone()]

        for i in range(planner.ddim_steps):
            t_batch   = planner.make_timesteps(B, int(planner.ddim_t[i].item()), device)
            idx_batch = planner.make_timesteps(B, i, device)

            mu, logvar, _ = planner.p_mean_variance(
                z, t_batch, idx_batch, vl_embeds, his_embeds, ego_embeds,
                deterministic=False,
            )
            sigma = (0.5 * logvar).exp().clamp(min=self.min_sigma).to(dtype)
            # Clip noise to avoid extreme samples
            noise = torch.randn_like(z).clamp_(-5.0, 5.0)
            z = (mu + sigma * noise).detach()
            chain.append(z.clone())

        return torch.stack(chain, dim=1)  # (B, K+1, H, D)

    def _compute_log_probs_old(
        self, planner, vl_embeds, his_embeds, ego_embeds, chain, B, K, device, dtype
    ):
        """
        Compute per-step log_prob under the current (snapshot) student policy.

        This is called inside no_grad, so no computation graph is built.

        Returns:
            log_probs_old (B, K)
        """
        lp = torch.zeros(B, K, device=device, dtype=dtype)
        for i in range(K):
            z_t   = chain[:, i].to(dtype)
            z_t1  = chain[:, i + 1].to(dtype)
            t_batch   = planner.make_timesteps(B, int(planner.ddim_t[i].item()), device)
            idx_batch = planner.make_timesteps(B, i, device)

            mu, logvar, _ = planner.p_mean_variance(
                z_t, t_batch, idx_batch, vl_embeds, his_embeds, ego_embeds,
                deterministic=False,
            )
            # Use a slightly larger min_sigma for log-prob stability
            sigma = (0.5 * logvar).exp().clamp(min=self.min_logprob_sigma).to(dtype)
            lp[:, i] = Normal(mu, sigma).log_prob(z_t1).sum(dim=(1, 2))
        return lp

    # ─────────────────────────────────────────────────────────────────────────
    # Main training entry point
    # ─────────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        student_planner,
        teacher_planner,
        vl_features: torch.Tensor,
        action_input,
    ) -> BatchFeature:
        """
        Compute the OPD loss for one training step.

        Args:
            student_planner: Trainable ReCogDriveDiffusionPlanner (IL-init).
            teacher_planner: Frozen  ReCogDriveDiffusionPlanner (RL-trained).
            vl_features:     Cached VLM last_hidden_state, shape (B, N, 1536).
            action_input:    BatchFeature with:
                               'his_traj'       (B, 12)
                               'status_feature' (B, 8)

        Returns:
            BatchFeature with scalar 'loss' and diagnostic keys.
        """
        B      = vl_features.shape[0]
        device = vl_features.device
        s_dtype = next(student_planner.parameters()).dtype
        t_dtype = next(teacher_planner.parameters()).dtype

        his_traj   = action_input.his_traj        # (B, 12)
        ego_status = action_input.status_feature  # (B, 8)

        # ── Phase 1 & 2: on-policy chain sampling + log_prob_old (no grad) ───
        with torch.no_grad():
            vl_s0, his_s0, ego_s0 = self._encode(
                student_planner, vl_features, his_traj, ego_status, s_dtype
            )
            chain = self._sample_chain(
                student_planner, vl_s0, his_s0, ego_s0, B, device, s_dtype
            )  # (B, K+1, H, D) — fully detached
            K = chain.shape[1] - 1
            log_probs_old = self._compute_log_probs_old(
                student_planner, vl_s0, his_s0, ego_s0,
                chain, B, K, device, s_dtype
            )  # (B, K)

        # ── Encode teacher features once (no grad) ────────────────────────────
        with torch.no_grad():
            vl_t, his_t, ego_t = self._encode(
                teacher_planner, vl_features, his_traj, ego_status, t_dtype
            )

        # ── Phase 3: student forward with grad + per-step PPO-clip loss ───────
        # Encode student features *with* grad so all student DiT params are trained.
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, s_dtype
        )

        total_loss = vl_features.new_zeros(())
        kl_per_step: list[torch.Tensor] = []

        for i in range(K):
            z_t  = chain[:, i].to(s_dtype)      # (B, H, D) — detached from Phase 1
            z_t1 = chain[:, i + 1].to(s_dtype)  # (B, H, D)

            t_batch   = student_planner.make_timesteps(B, int(student_planner.ddim_t[i].item()), device)
            idx_batch = student_planner.make_timesteps(B, i, device)

            # Student forward — gradient flows through mu_s only (sigma is fixed by eta)
            mu_s, logvar_s, _ = student_planner.p_mean_variance(
                z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False,
            )
            # Detach sigma: eta is a fixed parameter; gradients through sigma would
            # push it to infinity to artificially minimise KL.
            sigma_s = (0.5 * logvar_s).exp().clamp(min=self.min_sigma).to(s_dtype).detach()

            # log_prob_new for the importance-sampling ratio
            log_prob_new = (
                Normal(mu_s, sigma_s).log_prob(z_t1.detach()).sum(dim=(1, 2))
            )  # (B,)

            # Teacher mean — no gradient
            with torch.no_grad():
                mu_t, _, _ = teacher_planner.p_mean_variance(
                    z_t.to(t_dtype), t_batch, idx_batch,
                    vl_t, his_t, ego_t, deterministic=True,
                )
                mu_t = mu_t.to(s_dtype)

            # KL: ‖μ_s − μ_t‖² / (2σ_s²)  summed over waypoint dims
            kl = ((mu_s - mu_t.detach()).pow(2) / (2.0 * sigma_s.pow(2))).sum(dim=(1, 2))  # (B,)
            kl_per_step.append(kl.detach().mean())

            # Advantage = −KL (optionally zero-mean normalised over batch)
            if self.normalize_advantage and B > 1:
                adv = -(kl - kl.mean()) / (kl.std() + 1e-8)
            else:
                adv = -kl

            # PPO-clip objective
            ratio      = (log_prob_new - log_probs_old[:, i].detach()).exp()
            pg_loss    = -adv.detach() * ratio
            pg_clipped = -adv.detach() * ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip)
            step_loss  = torch.max(pg_loss, pg_clipped).mean()
            total_loss = total_loss + step_loss

        loss    = total_loss / K
        kl_mean = torch.stack(kl_per_step).mean()

        return BatchFeature(data={
            "loss":         loss,
            "kl_mean":      kl_mean,
            "distill_loss": kl_mean,   # reuse existing logging key
        })

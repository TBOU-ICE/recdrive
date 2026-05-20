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

IMPORTANT: log_prob_old and log_prob_new MUST use the same min_sigma so that
ratio ≈ 1 at initialisation. Using different clamp floors (e.g. min_logprob_sigma
≠ min_sigma) inflates the ratio by exp(Δ·D) where D is the action dimensionality,
causing immediate NaN when the DDIM sigma ≈ 0 at the first denoising step.
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
        normalize_advantage: bool = True,
    ):
        self.eps_clip = eps_clip
        self.min_sigma = min_sigma
        # Use the SAME floor for old and new policy so ratio ≈ 1 at init.
        self.min_logprob_sigma = min_sigma
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

    @staticmethod
    def _safe_sigma(logvar: torch.Tensor, min_sigma: float, dtype: torch.dtype) -> torch.Tensor:
        """Numerically-safe sigma: clamp logvar before exp to prevent overflow."""
        return (0.5 * logvar.float().clamp(-20.0, 20.0)).exp().clamp(min=min_sigma).to(dtype)

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
            sigma = self._safe_sigma(logvar, self.min_sigma, dtype)
            noise = torch.randn_like(z).clamp_(-5.0, 5.0)
            z = (mu + sigma * noise).detach()
            chain.append(z.clone())

        return torch.stack(chain, dim=1)  # (B, K+1, H, D)

    def _compute_log_probs_old(
        self, planner, vl_embeds, his_embeds, ego_embeds, chain, B, K, device, dtype
    ):
        """
        Compute per-step log_prob under the current (snapshot) student policy.

        Uses the SAME min_sigma as Phase 3 so that ratio ≈ 1 at initialisation.
        Called inside no_grad; no computation graph is built.

        Returns:
            log_probs_old (B, K)
        """
        lp = torch.zeros(B, K, device=device, dtype=torch.float32)
        for i in range(K):
            z_t   = chain[:, i].to(dtype)
            z_t1  = chain[:, i + 1].to(dtype)
            t_batch   = planner.make_timesteps(B, int(planner.ddim_t[i].item()), device)
            idx_batch = planner.make_timesteps(B, i, device)

            mu, logvar, _ = planner.p_mean_variance(
                z_t, t_batch, idx_batch, vl_embeds, his_embeds, ego_embeds,
                deterministic=False,
            )
            # Use the same min_sigma floor as log_prob_new so ratio≈1 at init.
            sigma = self._safe_sigma(logvar, self.min_logprob_sigma, dtype)
            lp[:, i] = Normal(mu.float(), sigma.float()).log_prob(z_t1.float()).sum(dim=(1, 2))
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
            )  # (B, K) in float32

        # ── Encode teacher features in float32 for precise reference targets ─
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            vl_t, his_t, ego_t = self._encode(
                teacher_planner, vl_features, his_traj, ego_status, torch.float32
            )

        # ── Phase 3: student forward with grad + per-step PPO-clip loss ───────
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, s_dtype
        )

        total_loss = vl_features.new_zeros(())
        kl_per_step:        list[torch.Tensor] = []
        ratio_per_step:     list[torch.Tensor] = []
        clip_frac_per_step: list[torch.Tensor] = []
        sigma_per_step:     list[torch.Tensor] = []
        logvar_per_step:    list[torch.Tensor] = []

        for i in range(K):
            z_t  = chain[:, i].to(s_dtype)
            z_t1 = chain[:, i + 1].to(s_dtype)

            t_batch   = student_planner.make_timesteps(B, int(student_planner.ddim_t[i].item()), device)
            idx_batch = student_planner.make_timesteps(B, i, device)

            # Student forward — gradient flows through mu_s only (sigma detached).
            mu_s, logvar_s, _ = student_planner.p_mean_variance(
                z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False,
            )
            sigma_s = self._safe_sigma(logvar_s, self.min_sigma, s_dtype).detach()

            # log_prob_new — same sigma formula as log_probs_old.
            log_prob_new = (
                Normal(mu_s.float(), sigma_s.float())
                .log_prob(z_t1.detach().float())
                .sum(dim=(1, 2))
            )  # (B,) in float32

            # Teacher mean in float32 — no gradient.
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                mu_t, _, _ = teacher_planner.p_mean_variance(
                    z_t.float(), t_batch, idx_batch,
                    vl_t, his_t, ego_t, deterministic=True,
                )

            # KL: ‖μ_s − μ_t‖² / (2σ_s²) in float32 for stability.
            kl = (
                (mu_s.float() - mu_t.detach())
                .pow(2)
                .div(2.0 * sigma_s.float().pow(2))
                .sum(dim=(1, 2))
            )  # (B,)
            kl_per_step.append(kl.detach().mean())

            # Advantage = −KL (optionally zero-mean normalised over batch).
            if self.normalize_advantage and B > 1:
                adv = -(kl - kl.mean()) / (kl.std() + 1e-8)
            else:
                adv = -kl

            # PPO-clip: clamp log-ratio to [-5, 5] before exp for stability.
            log_ratio = (log_prob_new - log_probs_old[:, i].detach()).clamp(-5.0, 5.0)
            ratio = log_ratio.exp()

            pg_loss    = -adv.detach() * ratio
            pg_clipped = -adv.detach() * ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip)
            step_loss  = torch.max(pg_loss, pg_clipped).mean()
            total_loss = total_loss + step_loss.to(total_loss.dtype)

            # Diagnostics (detached, no graph).
            ratio_per_step.append(ratio.detach().mean())
            clip_frac_per_step.append(((ratio.detach() - 1.0).abs() > self.eps_clip).float().mean())
            sigma_per_step.append(sigma_s.detach().float().mean())
            logvar_per_step.append(logvar_s.detach().float().mean())

        loss    = total_loss / K
        kl_mean = torch.stack(kl_per_step).mean()

        # Guard: if loss is non-finite, zero it out and log a warning.
        if not torch.isfinite(loss):
            loss = loss.new_zeros(())

        ratio_mean = torch.stack(ratio_per_step).mean()
        clip_frac  = torch.stack(clip_frac_per_step).mean()
        sigma_mean  = torch.stack(sigma_per_step).mean()
        logvar_mean = torch.stack(logvar_per_step).mean()
        chain_abs_max = chain.float().abs().max()

        return BatchFeature(data={
            "loss":           loss,
            "kl_mean":        kl_mean,
            "distill_loss":   kl_mean,
            "ratio_mean":     ratio_mean,
            "ratio_clip_frac": clip_frac,
            "sigma_mean":     sigma_mean,
            "logvar_mean":    logvar_mean,
            "chain_abs_max":  chain_abs_max,
        })

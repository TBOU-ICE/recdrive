"""
DiT OPD (On-Policy Distillation) trainer for ReCogDrive.

Implements Flow-OPD (arXiv:2605.08063) adapted to the DDIM denoising setting.

Core algorithm (Flow-OPD Eq. 8-10):
  For each denoising step i, evaluate both student and teacher at the SAME
  state z_t drawn from the student's OWN stochastic chain.  The loss is the
  σ-weighted forward KL between student and teacher denoising distributions:

      KL_i = ‖μ_θ(z_t, t) − μ_φ(z_t, t)‖² / (2 σ_i²)

  σ_i is the DDIM step noise (schedule-determined).  This w(t) weighting
  (Flow-OPD Eq. 10) down-weights noisy steps (large σ) and up-weights clean
  steps (small σ, close to the final trajectory).

  Total loss = (1/K) Σ_i mean_batch(KL_i)

Why no PPO-clip:
  PPO requires a frozen π_θ_old held fixed across MULTIPLE gradient updates so
  that the importance ratio ρ = π_θ/π_θ_old deviates from 1.  When the chain
  is re-sampled every step (old policy ≡ current policy), ρ ≡ 1 always, the
  PPO surrogate collapses to mean(-adv)=0 with a zero-mean normalised advantage
  — confirmed by TensorBoard showing loss≡0, ratio≡1, clip_frac≡0 throughout
  training.  Direct on-policy KL avoids this degenerate case.

Diagnostic logging (every log_interval calls, rank-0 only):
  Writes a human-readable table to <log_dir>/distill_diagnostic.log showing
  per-step σ, ‖μ_θ−μ_φ‖, KL, weight, and one example denormalized trajectory
  comparison (teacher vs student at the final DDIM step).
"""

import os
import datetime
import torch
from transformers.feature_extraction_utils import BatchFeature


class ReCogDriveDiTDistillTrainer:
    """
    Flow-OPD-style on-policy KL distillation between two DiT planners.

    Directly minimises KL(student ‖ teacher) at states sampled from the
    student's own stochastic DDIM chain.
    """

    def __init__(
        self,
        eps_clip: float = 0.2,            # kept for API compatibility, unused
        min_sigma: float = 0.04,
        normalize_advantage: bool = True, # kept for API compatibility, unused
        log_dir: str = None,
        log_interval: int = 50,
    ):
        self.min_sigma  = min_sigma
        self.log_dir    = log_dir
        self.log_interval = log_interval
        self._call_count  = 0
        self._local_rank  = int(os.getenv("LOCAL_RANK", "0"))

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _encode(planner, vl_features, his_traj, ego_status, dtype):
        """Encode raw conditioning tensors through the planner's projections."""
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
        Student on-policy stochastic DDIM chain (no_grad).

        Uses EtaFixed ≈ 1.0 for exploration (SDE sampling in Flow-OPD).

        Returns:
            chain (B, K+1, H, D) — all detached
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

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostic logging
    # ─────────────────────────────────────────────────────────────────────────

    def _write_text_log(
        self,
        step: int,
        K: int,
        ddim_t_list,
        kl_weighted_per_step,   # list[float] — KL_i averaged over batch
        kl_raw_per_step,        # list[float] — ||mu_s-mu_t||^2 / dim, averaged over batch
        sigma_per_step,         # list[float]
        mu_diff_l2_per_step,    # list[float] — ||mu_s-mu_t||_F / sqrt(H*D)
        mu_s_last: torch.Tensor,  # (B, H, D) student mean at final DDIM step
        mu_t_last: torch.Tensor,  # (B, H, D) teacher mean at final DDIM step
        student_planner,
        total_loss: float,
    ):
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "distill_diagnostic.log")

        # Denormalize example trajectories (sample 0)
        with torch.no_grad():
            mu_s_denorm = student_planner.denorm_odo(mu_s_last[:1].float().cpu())  # (1, H, 3)
            mu_t_denorm = student_planner.denorm_odo(mu_t_last[:1].float().cpu())  # (1, H, 3)

        ts = [0.5 * (i + 1) for i in range(mu_s_denorm.shape[1])]  # 0.5..4.0s

        lines = []
        sep = "=" * 78
        lines.append(f"\n{sep}")
        lines.append(
            f"[DiT-OPD step={step:>6d}]  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"  K={K} DDIM steps | total_loss={total_loss:.4f}")
        lines.append("-" * 78)
        lines.append(
            f"{'idx':>3} {'t':>4} {'σ_i':>7} {'‖μ_θ-μ_φ‖/dim':>14} "
            f"{'1/(2σ²)':>10} {'KL_i':>10} {'share%':>7}"
        )
        lines.append("-" * 78)
        total_kl = sum(kl_weighted_per_step)
        for i in range(K):
            weight = 1.0 / (2.0 * sigma_per_step[i] ** 2 + 1e-12)
            share = 100.0 * kl_weighted_per_step[i] / (total_kl + 1e-12)
            lines.append(
                f"  {i:>1}  {ddim_t_list[i]:>3}  {sigma_per_step[i]:>7.4f}  "
                f"{kl_raw_per_step[i]:>14.4f}  "
                f"{weight:>10.2f}  {kl_weighted_per_step[i]:>10.3f}  {share:>6.1f}%"
            )
        lines.append("-" * 78)
        lines.append(f"  mean KL over {K} steps = {total_loss:.4f}")
        lines.append("-" * 78)

        # Example trajectory at final DDIM step
        lines.append("  Sample-0 final step (t→0) predictions [denormalized x/y/heading(rad)]:")
        lines.append(f"  {'time':>6}  {'Teacher μ_φ':>32}  {'Student μ_θ':>32}")
        for j, t_sec in enumerate(ts):
            tx, ty, th = (
                mu_t_denorm[0, j, 0].item(),
                mu_t_denorm[0, j, 1].item(),
                mu_t_denorm[0, j, 2].item(),
            )
            sx, sy, sh = (
                mu_s_denorm[0, j, 0].item(),
                mu_s_denorm[0, j, 1].item(),
                mu_s_denorm[0, j, 2].item(),
            )
            lines.append(
                f"  {t_sec:>5.1f}s  "
                f"({tx:>7.3f}, {ty:>7.3f}, {th:>7.4f})   "
                f"({sx:>7.3f}, {sy:>7.3f}, {sh:>7.4f})"
            )
        lines.append(sep)

        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")

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
        Compute the on-policy KL distillation loss for one training step.

        Args:
            student_planner: Trainable ReCogDriveDiffusionPlanner (IL-init).
            teacher_planner: Frozen  ReCogDriveDiffusionPlanner (RL-trained).
            vl_features:     Cached VLM last_hidden_state, shape (B, N, 1536).
            action_input:    BatchFeature with 'his_traj' (B,12) and
                             'status_feature' (B,8).

        Returns:
            BatchFeature with scalar 'loss' and diagnostic keys for TensorBoard.
        """
        self._call_count += 1

        B       = vl_features.shape[0]
        device  = vl_features.device
        s_dtype = next(student_planner.parameters()).dtype

        his_traj   = action_input.his_traj
        ego_status = action_input.status_feature

        # ── Phase 1: student on-policy chain (no grad) ────────────────────────
        with torch.no_grad():
            vl_s0, his_s0, ego_s0 = self._encode(
                student_planner, vl_features, his_traj, ego_status, s_dtype
            )
            chain = self._sample_chain(
                student_planner, vl_s0, his_s0, ego_s0, B, device, s_dtype
            )  # (B, K+1, H, D) — fully detached
        K = chain.shape[1] - 1

        # ── Teacher encode (no grad, float32 for reference) ───────────────────
        with torch.no_grad():
            vl_t, his_t, ego_t = self._encode(
                teacher_planner, vl_features, his_traj, ego_status, torch.float32
            )

        # ── Phase 2: student forward with grad + per-step KL ──────────────────
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, s_dtype
        )

        total_loss         = vl_features.new_zeros(())
        kl_weighted_list:  list[torch.Tensor] = []   # after σ weighting, for loss
        kl_raw_list:       list[torch.Tensor] = []   # ‖μ_s-μ_t‖²/dim, for diagnostics
        mu_diff_l2_list:   list[torch.Tensor] = []   # ‖μ_s-μ_t‖_F / sqrt(H*D)
        sigma_list:        list[torch.Tensor] = []

        # Save last-step μ for text log
        mu_s_last = mu_t_last = None

        ddim_t_list = [int(student_planner.ddim_t[i].item()) for i in range(K)]

        for i in range(K):
            z_t = chain[:, i].to(s_dtype)

            t_batch   = student_planner.make_timesteps(B, ddim_t_list[i], device)
            idx_batch = student_planner.make_timesteps(B, i, device)

            # Student forward — gradient flows through μ_θ.
            mu_s, logvar_s, _ = student_planner.p_mean_variance(
                z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False,
            )
            # σ_i: DDIM schedule noise (no learnable dependency, detach for safety).
            sigma_s = self._safe_sigma(logvar_s, self.min_sigma, s_dtype).detach()

            # Teacher mean — deterministic prediction, no gradient.
            with torch.no_grad():
                mu_t, _, _ = teacher_planner.p_mean_variance(
                    z_t.float(), t_batch, idx_batch,
                    vl_t, his_t, ego_t, deterministic=True,
                )

            # Forward KL (Gaussian, shared σ_i):
            #   KL(N(μ_θ, σ_i²I) ‖ N(μ_φ, σ_i²I)) = ‖μ_θ − μ_φ‖² / (2σ_i²)
            #
            # σ-weighting (Flow-OPD Eq. 10):
            #   noisy steps (large σ) → small weight  (coarse structural correction)
            #   clean steps (small σ) → large weight  (fine trajectory matching)
            diff = mu_s.float() - mu_t.detach()                            # (B, H, D)
            sigma2 = sigma_s.float().pow(2).clamp(min=1e-6)                # scalar-ish
            kl_i   = diff.pow(2).div(2.0 * sigma2).sum(dim=(1, 2))        # (B,)

            step_loss  = kl_i.mean()
            total_loss = total_loss + step_loss.to(total_loss.dtype)

            H, D = diff.shape[1], diff.shape[2]
            kl_weighted_list.append(kl_i.detach().mean())
            kl_raw_list.append((diff.pow(2).sum(dim=(1, 2)) / (H * D)).detach().mean())
            mu_diff_l2_list.append(
                (diff.pow(2).sum(dim=(1, 2)).sqrt() / (H * D) ** 0.5).detach().mean()
            )
            sigma_list.append(sigma_s.detach().float().mean())

            if i == K - 1:
                mu_s_last = mu_s.detach()
                mu_t_last = mu_t.detach()

        loss    = total_loss / K
        kl_mean = torch.stack(kl_weighted_list).mean()

        if not torch.isfinite(loss):
            loss = loss.new_zeros(())

        # ── Trajectory comparison metrics ─────────────────────────────────────
        kl_steps = torch.stack(kl_weighted_list)  # (K,)
        with torch.no_grad():
            pred_traj_s = student_planner.denorm_odo(mu_s_last.float().cpu()).to(device)  # (B, H, 3)
            pred_traj_t = student_planner.denorm_odo(mu_t_last.float().cpu()).to(device)  # (B, H, 3)
            pred_traj_l1 = torch.nn.functional.l1_loss(pred_traj_s, pred_traj_t)

        # ── Diagnostic text log (rank-0, every log_interval calls) ────────────
        if (
            self.log_dir is not None
            and self._call_count % self.log_interval == 0
            and self._local_rank == 0
            and mu_s_last is not None
        ):
            self._write_text_log(
                step=self._call_count,
                K=K,
                ddim_t_list=ddim_t_list,
                kl_weighted_per_step=[v.item() for v in kl_weighted_list],
                kl_raw_per_step=[v.item() for v in kl_raw_list],
                sigma_per_step=[v.item() for v in sigma_list],
                mu_diff_l2_per_step=[v.item() for v in mu_diff_l2_list],
                mu_s_last=mu_s_last,
                mu_t_last=mu_t_last,
                student_planner=student_planner,
                total_loss=loss.item(),
            )

        # ── TensorBoard scalars ────────────────────────────────────────────────
        # Per-step KL (weighted) for detailed analysis
        per_step_kl = {
            f"kl_step_{i}": kl_weighted_list[i] for i in range(K)
        }
        per_step_raw = {
            f"kl_raw_step_{i}": kl_raw_list[i] for i in range(K)
        }
        per_step_sigma = {
            f"sigma_step_{i}": sigma_list[i] for i in range(K)
        }

        return BatchFeature(data={
            "loss":                           loss,
            "kl_mean":                        kl_mean,
            "distill_loss":                   kl_mean,
            "transition_kl":                  kl_mean.detach(),
            "step_kl_mean":                   kl_steps.mean().detach(),
            "step_kl_max":                    kl_steps.max().detach(),
            "step_kls":                       kl_steps.detach(),
            "sigma_mean":                     torch.stack(sigma_list).mean(),
            "chain_abs_max":                  chain.float().abs().max(),
            "denoising_steps":                torch.tensor(float(K), device=device),
            "pred_traj_l1_to_teacher":        pred_traj_l1.detach(),
            "student_pred_traj_mean":         pred_traj_s.mean().detach(),
            "student_pred_traj_std":          pred_traj_s.std(unbiased=False).detach(),
            "teacher_pred_traj_mean":         pred_traj_t.mean().detach(),
            "teacher_pred_traj_std":          pred_traj_t.std(unbiased=False).detach(),
            "teacher_student_pred_traj_abs_mean": (pred_traj_s - pred_traj_t).abs().mean().detach(),
            "student_pred_traj_first_point":  pred_traj_s[0, 0].detach(),
            "teacher_pred_traj_first_point":  pred_traj_t[0, 0].detach(),
            **per_step_kl,
            **per_step_raw,
            **per_step_sigma,
        })

"""
Scene-router four-teacher DiT OPD distillation for ReCogDrive (v1).

Additive file: does not modify any existing implementation.

v1 design (all knobs are configurable):
- On-policy student DDIM chain (detached rollout), same as the fixed-weight
  four-teacher OPD trainer.
- ``teacher_select='scene_route'``: every sample is matched to ONLY the teacher
  of its scenario bucket (progress / rule / safety / general), instead of a
  fixed-weight average over all four teachers. This removes the "mode averaging"
  failure where averaging multi-modal expert trajectories yields an off-manifold
  compromise.
- ``match_target in {'mu', 'x0'}``: match either the DDIM posterior transition
  mean ``mu`` (the original formulation) or the predicted clean trajectory
  ``x_recon`` (eta-independent, better scaled). Still on-policy OPD; only the
  regression-target parameterization changes. Precision weighting ``1/(2 sigma^2)``
  is kept for both.
- ``exopd_lambda`` (ExOPD reward extrapolation, G-OPD): target is extrapolated
  beyond the expert along ``(expert - ref)``:
      target = ref + lambda * (expert - ref)
  with ``ref`` a frozen IL-base planner (the experts' pre-RL base). ``lambda=1.0``
  disables extrapolation (target == expert). In the shared-variance Gaussian case
  this is exactly the ExOPD optimum in mean/x0 space.
"""

from typing import Dict, List, Optional

import torch
from transformers.feature_extraction_utils import BatchFeature


class ReCogDriveDiTSceneRouterDistillTrainer:
    """Scene-routed (optionally reward-extrapolated) OPD distillation."""

    def __init__(
        self,
        bucket_names: List[str],
        fallback_bucket: str = "general_or_no_tag",
        min_sigma: float = 0.04,
        smooth_weight: float = 0.02,
        match_target: str = "x0",
        exopd_lambda: float = 1.0,
    ):
        if not bucket_names:
            raise ValueError("bucket_names must not be empty")
        if match_target not in ("mu", "x0"):
            raise ValueError(f"match_target must be 'mu' or 'x0', got {match_target!r}")
        self.bucket_names: List[str] = list(bucket_names)
        self.fallback_bucket = fallback_bucket
        self.min_sigma = float(min_sigma)
        self.smooth_weight = float(smooth_weight)
        self.match_target = match_target
        self.exopd_lambda = float(exopd_lambda)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _encode(planner, vl_features, his_traj, ego_status, dtype):
        vl_embeds = planner.feature_encoder(vl_features.to(dtype))
        his_embeds = (
            planner.his_traj_encoder(his_traj.to(dtype).unsqueeze(1))
            .repeat(1, planner.config.action_horizon, 1)
        )
        ego_embeds = planner.ego_status_encoder(ego_status.to(dtype))
        return vl_embeds, his_embeds, ego_embeds

    @staticmethod
    def _safe_sigma(logvar: torch.Tensor, min_sigma: float, dtype: torch.dtype) -> torch.Tensor:
        return (0.5 * logvar.float().clamp(-20.0, 20.0)).exp().clamp(min=min_sigma).to(dtype)

    @staticmethod
    def _jerk_loss(traj: torch.Tensor) -> torch.Tensor:
        if traj.shape[1] < 4:
            return traj.new_zeros(())
        xy = traj[..., :2]
        jerk = xy[:, 3:] - 3.0 * xy[:, 2:-1] + 3.0 * xy[:, 1:-2] - xy[:, :-3]
        return jerk.pow(2).mean()

    def _sample_chain(self, planner, vl_embeds, his_embeds, ego_embeds, batch_size, device, dtype):
        """Detached stochastic DDIM chain sampled from the current student."""
        horizon = planner.config.action_horizon
        action_dim = planner.config.action_dim
        z = torch.randn((batch_size, horizon, action_dim), device=device, dtype=dtype)
        chain = [z.clone()]
        for i in range(planner.ddim_steps):
            t_batch = planner.make_timesteps(batch_size, int(planner.ddim_t[i].item()), device)
            idx_batch = planner.make_timesteps(batch_size, i, device)
            mu, logvar, _ = planner.p_mean_variance(
                z, t_batch, idx_batch, vl_embeds, his_embeds, ego_embeds, deterministic=False
            )
            sigma = self._safe_sigma(logvar, self.min_sigma, dtype)
            noise = torch.randn_like(z).clamp_(-5.0, 5.0)
            z = (mu + sigma * noise).detach()
            chain.append(z.clone())
        return torch.stack(chain, dim=1)

    def _resolve_bucket(self, name: str) -> str:
        """Map a per-sample bucket name to an available teacher key."""
        if name in self.bucket_names:
            return name
        return self.fallback_bucket

    # ------------------------------------------------------------------- loss
    def compute_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        ref_planner: Optional[torch.nn.Module],
        vl_features: torch.Tensor,
        action_input,
        bucket_per_sample: List[str],
    ) -> BatchFeature:
        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature

        use_exopd = ref_planner is not None and abs(self.exopd_lambda - 1.0) > 1e-6
        use_x0 = self.match_target == "x0"

        # 1) on-policy detached student DDIM chain
        with torch.no_grad():
            vl_s0, his_s0, ego_s0 = self._encode(
                student_planner, vl_features, his_traj, ego_status, student_dtype
            )
            chain = self._sample_chain(
                student_planner, vl_s0, his_s0, ego_s0, batch_size, device, student_dtype
            )
        num_steps = chain.shape[1] - 1
        ddim_t_list = [int(student_planner.ddim_t[i].item()) for i in range(num_steps)]

        # 2) teacher / ref encodings (frozen, float32)
        teacher_encodings: Dict[str, tuple] = {}
        with torch.no_grad():
            for name, teacher in teacher_planners.items():
                teacher.eval()
                teacher_encodings[name] = self._encode(
                    teacher, vl_features, his_traj, ego_status, torch.float32
                )
            ref_encoding = (
                self._encode(ref_planner, vl_features, his_traj, ego_status, torch.float32)
                if use_exopd else None
            )

        # 3) student encodings (with grad)
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, student_dtype
        )

        # per-sample -> resolved bucket, grouped indices
        resolved = [self._resolve_bucket(b) for b in bucket_per_sample]
        bucket_to_indices: Dict[str, List[int]] = {}
        for i, b in enumerate(resolved):
            bucket_to_indices.setdefault(b, []).append(i)

        total_loss = vl_features.new_zeros(())
        per_bucket_step_losses: Dict[str, list] = {b: [] for b in self.bucket_names}
        sigma_list = []
        last_student_x0 = None

        for step in range(num_steps):
            z_t = chain[:, step].to(student_dtype)
            t_batch = student_planner.make_timesteps(batch_size, ddim_t_list[step], device)
            idx_batch = student_planner.make_timesteps(batch_size, step, device)

            mu_s, logvar_s, x0_s = student_planner.p_mean_variance(
                z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False
            )
            sigma_s = self._safe_sigma(logvar_s, self.min_sigma, student_dtype).detach()
            sigma2 = sigma_s.float().pow(2).clamp(min=1e-6)
            sigma_list.append(sigma_s.detach().float().mean())
            student_side = (x0_s if use_x0 else mu_s).float()

            # ref target once per step on full batch (bucket-independent)
            ref_side_full = None
            if use_exopd:
                with torch.no_grad():
                    mu_r, _, x0_r = ref_planner.p_mean_variance(
                        z_t.float(), t_batch, idx_batch,
                        ref_encoding[0], ref_encoding[1], ref_encoding[2],
                        deterministic=True,
                    )
                    ref_side_full = (x0_r if use_x0 else mu_r).detach()

            step_loss = vl_features.new_zeros(())
            n_used = 0
            for bucket, indices in bucket_to_indices.items():
                enc = teacher_encodings[bucket]
                teacher = teacher_planners[bucket]
                sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                z_sel = z_t[sel].float()
                t_sel = t_batch[sel]
                idx_sel = idx_batch[sel]
                with torch.no_grad():
                    mu_t, _, x0_t = teacher.p_mean_variance(
                        z_sel, t_sel, idx_sel,
                        enc[0][sel], enc[1][sel], enc[2][sel],
                        deterministic=True,
                    )
                    teacher_side = (x0_t if use_x0 else mu_t).detach()
                    if use_exopd:
                        ref_side = ref_side_full[sel]
                        target = ref_side + self.exopd_lambda * (teacher_side - ref_side)
                    else:
                        target = teacher_side

                diff = student_side[sel] - target
                loss_per_sample = diff.pow(2).div(2.0 * sigma2[sel]).sum(dim=(1, 2))
                per_bucket_step_losses[bucket].append(loss_per_sample.mean().detach())
                step_loss = step_loss + loss_per_sample.sum()
                n_used += len(indices)

            total_loss = total_loss + step_loss / max(n_used, 1)
            if step == num_steps - 1:
                last_student_x0 = x0_s

        distill_loss = total_loss / num_steps
        pred_traj_s = student_planner.denorm_odo(last_student_x0.float())
        smooth_loss = self._jerk_loss(pred_traj_s)
        loss = distill_loss + self.smooth_weight * smooth_loss
        # Under match_target=x0, sigma/eta is detached from the regression target, so
        # eta_logit would be an unused DDP parameter. Keep it in the graph with a
        # zero coefficient so ranks stay collective-symmetric without find_unused.
        # NB: eta_logit is atanh(1.0)=+inf when base_eta==max_eta (EtaFixed) and that
        # inf lives in the student checkpoint, so a plain ``* 0.0`` gives inf*0=NaN and
        # trips the finite-guard below every step (student silently never trains).
        # nan_to_num keeps the graph edge while contributing a guaranteed finite zero.
        if hasattr(student_planner, "eta") and hasattr(student_planner.eta, "eta_logit"):
            eta_logit = student_planner.eta.eta_logit
            loss = loss + torch.nan_to_num(eta_logit, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        if not torch.isfinite(loss):
            # Keep a live grad graph (zeros(()) would skip DDP reductions on this rank).
            loss = last_student_x0.float().sum() * 0.0

        data = {
            "loss": loss,
            "distill_loss": distill_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
            "weighted_smooth_loss": (self.smooth_weight * smooth_loss).detach(),
            "sigma_mean": torch.stack(sigma_list).mean().detach(),
            "chain_abs_max": chain.float().abs().max().detach(),
            "denoising_steps": torch.tensor(float(num_steps), device=device),
            "exopd_lambda": torch.tensor(float(self.exopd_lambda), device=device),
            "student_pred_traj_mean": pred_traj_s.mean().detach(),
            "student_pred_traj_std": pred_traj_s.std(unbiased=False).detach(),
        }
        # Always emit ALL buckets' keys (0 when absent): the lightning module logs
        # each key with sync_dist=True, so a rank-dependent key set desynchronizes
        # NCCL collectives across DDP ranks and deadlocks training silently.
        for bucket in self.bucket_names:
            values = per_bucket_step_losses[bucket]
            data[f"kl_{bucket}_mean"] = (
                torch.stack(values).mean().detach()
                if values
                else torch.tensor(0.0, device=device)
            )
            data[f"n_samples_{bucket}"] = torch.tensor(
                float(len(bucket_to_indices.get(bucket, []))), device=device
            )
        return BatchFeature(data=data)

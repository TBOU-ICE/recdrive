"""
Pure OPRD-style multi-teacher DiT representation distillation trainer.

This file is additive: it does not modify the original temporal multi-teacher
OPD trainer or the original diffusion planner.  It reuses the existing student
on-policy DDIM chain, but removes trajectory / mu-level KL distillation and
uses only DiT hidden-state representation losses.

Design:
  - Student samples an on-policy denoising chain z_t with no gradient.
  - Student, IL teacher, and RL teacher are evaluated on the same z_t.
  - Middle denoising steps + middle DiT layers: align student to IL teacher.
  - Late denoising steps + last DiT layers: align student to IL-heavy/RL-light
    teacher representations.
  - No trajectory loss, no temporal consistency loss, no teacher gradients.
"""

from __future__ import annotations

import datetime
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


class ReCogDriveDiTOPRDMultiTeacherTrainer:
    """Pure hidden-state multi-teacher OPRD for ReCogDrive DiT planner."""

    def __init__(
        self,
        min_sigma: float = 0.04,
        log_dir: str | None = None,
        log_interval: int = 50,
        # Middle-step/middle-layer IL-only branch.
        mid_il_weight: float = 1.0,
        # Late-step/last-layer multi-teacher branch.
        last_il_weight: float = 0.85,
        last_rl_weight: float = 0.15,
        # Optional final DiT representation, i.e. before action_decoder.
        final_repr_weight: float = 1.0,
        use_final_repr: bool = True,
        # Normalization / loss settings.
        normalize_hidden: bool = True,
        loss_type: str = "mse",  # mse | smooth_l1
        # Layer and denoising-step partition ratios.
        mid_layer_start_ratio: float = 1.0 / 3.0,
        mid_layer_end_ratio: float = 2.0 / 3.0,
        last_layer_start_ratio: float = 2.0 / 3.0,
        middle_step_start_ratio: float = 1.0 / 3.0,
        late_step_start_ratio: float = 2.0 / 3.0,
    ) -> None:
        self.min_sigma = float(min_sigma)
        self.log_dir = log_dir
        self.log_interval = int(log_interval)
        self.mid_il_weight = float(mid_il_weight)
        self.last_il_weight = float(last_il_weight)
        self.last_rl_weight = float(last_rl_weight)
        self.final_repr_weight = float(final_repr_weight)
        self.use_final_repr = bool(use_final_repr)
        self.normalize_hidden = bool(normalize_hidden)
        self.loss_type = str(loss_type).lower()

        self.mid_layer_start_ratio = float(mid_layer_start_ratio)
        self.mid_layer_end_ratio = float(mid_layer_end_ratio)
        self.last_layer_start_ratio = float(last_layer_start_ratio)
        self.middle_step_start_ratio = float(middle_step_start_ratio)
        self.late_step_start_ratio = float(late_step_start_ratio)

        self._call_count = 0
        self._local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # ──────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────────────

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

    def _sample_chain(self, planner, vl_embeds, his_embeds, ego_embeds, B, device, dtype):
        """Student on-policy stochastic DDIM chain. Returns (B, K+1, H, D)."""
        H, D = planner.config.action_horizon, planner.config.action_dim
        z = torch.randn((B, H, D), device=device, dtype=dtype)
        chain = [z.clone()]

        for i in range(planner.ddim_steps):
            t_batch = planner.make_timesteps(B, int(planner.ddim_t[i].item()), device)
            idx_batch = planner.make_timesteps(B, i, device)
            mu, logvar, _ = planner.p_mean_variance(
                z, t_batch, idx_batch, vl_embeds, his_embeds, ego_embeds,
                deterministic=False,
            )
            sigma = self._safe_sigma(logvar, self.min_sigma, dtype)
            noise = torch.randn_like(z).clamp_(-5.0, 5.0)
            z = (mu + sigma * noise).detach()
            chain.append(z.clone())

        return torch.stack(chain, dim=1)

    @staticmethod
    def _range_from_ratios(
        total: int,
        start_ratio: float,
        end_ratio: float | None = None,
        one_based: bool = True,
    ) -> List[int]:
        """Return selected layer/step indices.

        For DiT hidden_states, one_based=True selects from [1, total], where
        hidden_states[0] is the input embedding and hidden_states[1:] are block
        outputs.  For denoising steps, one_based=False selects [0, total-1].
        """
        if total <= 0:
            return []
        start_ratio = max(0.0, min(1.0, float(start_ratio)))
        if end_ratio is None:
            end_ratio = 1.0
        end_ratio = max(start_ratio, min(1.0, float(end_ratio)))

        if one_based:
            start = int(total * start_ratio) + 1
            end = int(total * end_ratio)
            start = max(1, min(total, start))
            end = max(start, min(total, end))
            return list(range(start, end + 1))

        start = int(total * start_ratio)
        end = int(total * end_ratio) - 1
        start = max(0, min(total - 1, start))
        end = max(start, min(total - 1, end))
        return list(range(start, end + 1))

    def _select_indices(self, num_blocks: int, num_steps: int) -> Dict[str, List[int]]:
        mid_layers = self._range_from_ratios(
            num_blocks, self.mid_layer_start_ratio, self.mid_layer_end_ratio, one_based=True
        )
        last_layers = self._range_from_ratios(
            num_blocks, self.last_layer_start_ratio, 1.0, one_based=True
        )
        middle_steps = self._range_from_ratios(
            num_steps, self.middle_step_start_ratio, self.late_step_start_ratio, one_based=False
        )
        late_steps = self._range_from_ratios(
            num_steps, self.late_step_start_ratio, 1.0, one_based=False
        )
        return {
            "mid_layers": mid_layers,
            "last_layers": last_layers,
            "middle_steps": middle_steps,
            "late_steps": late_steps,
        }

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_hidden:
            return x.float()
        return F.normalize(x.float(), p=2, dim=-1, eps=1e-6)

    def _repr_distance(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = self._normalize(student)
        t = self._normalize(teacher.detach())
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(s, t)
        return F.mse_loss(s, t)

    def _p_mean_variance_with_repr(
        self,
        planner,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        deterministic: bool = True,
    ):
        """Same as ReCogDriveDiffusionPlanner.p_mean_variance, but returns DiT reps.

        This helper intentionally lives in the new trainer instead of modifying
        the original planner file.
        Returns:
            model_mean, model_log_variance, x_recon, model_output, hidden_states
        """
        model_dtype = next(planner.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = planner.action_encoder(x, t)
        if hasattr(planner, "position_embedding"):
            pos_ids = torch.arange(action_features.shape[1], device=x.device)
            action_features = action_features + planner.position_embedding(pos_ids)

        vl_features_mean = vl_features.mean(1).unsqueeze(1).repeat(1, planner.config.action_horizon, 1)
        fused_input = planner.fusion_projector(
            torch.cat((his_traj_features, vl_features_mean, action_features), dim=2)
        )

        model_output, hidden_states = planner.model(
            hidden_states=fused_input,
            encoder_hidden_states=vl_features,
            conditioning_features=ego_status_features,
            timesteps=t,
            return_hidden_states=True,
        )
        pred_noise = planner.action_decoder(model_output)

        if planner.config.sampling_method == "ddpm":
            x_recon = planner.extract(planner.ddpm_sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                planner.extract(planner.ddpm_sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
        elif planner.config.sampling_method == "ddim":
            alpha_t = planner.extract(planner.ddim_alphas, index, x.shape)
            sqrt_one_minus_alpha_t = planner.extract(planner.ddim_sqrt_one_minus_alphas, index, x.shape)
            x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t ** 0.5)
        else:
            raise ValueError(f"p_mean_variance not supported for method: {planner.config.sampling_method}")

        denoised_clip_value = getattr(planner, "denoised_clip_value", 1.0)
        x_recon = x_recon.clamp(-denoised_clip_value, denoised_clip_value)

        if planner.config.sampling_method == "ddpm":
            model_mean = planner.extract(planner.ddpm_mu_coef1, t, x.shape) * x_recon + \
                planner.extract(planner.ddpm_mu_coef2, t, x.shape) * x
            model_log_variance = planner.extract(planner.ddpm_logvar_clipped, t, x.shape)
        elif planner.config.sampling_method == "ddim":
            alpha_prev = planner.extract(planner.ddim_alphas_prev, index, x.shape)
            pred_noise = (x - (alpha_t ** 0.5) * x_recon) / sqrt_one_minus_alpha_t
            eps_clip_value = getattr(planner, "eps_clip_value", None)
            if eps_clip_value is not None:
                pred_noise = pred_noise.clamp(-eps_clip_value, eps_clip_value)

            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1), device=x.device, dtype=x.dtype)
            else:
                etas = planner.eta(x).unsqueeze(1)

            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            ).clamp(min=1e-10)
            pred_dir_xt = (1.0 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise
            model_mean = (alpha_prev ** 0.5) * x_recon + pred_dir_xt
            model_log_variance = torch.log(sigma ** 2 + 1e-20)

        return model_mean, model_log_variance, x_recon, model_output, hidden_states

    def _mean_layer_loss(self, hs_s: Sequence[torch.Tensor], hs_t: Sequence[torch.Tensor], layers: Iterable[int]) -> torch.Tensor:
        losses = []
        max_idx = min(len(hs_s), len(hs_t)) - 1
        for layer_idx in layers:
            if 0 <= layer_idx <= max_idx:
                losses.append(self._repr_distance(hs_s[layer_idx], hs_t[layer_idx]))
        if not losses:
            return hs_s[-1].new_zeros(())
        return torch.stack([v.to(hs_s[-1].dtype) for v in losses]).mean()

    def _write_text_log(self, step: int, indices: Dict[str, List[int]], payload: Dict[str, float]) -> None:
        if self.log_dir is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "oprd_multi_teacher_diagnostic.log")
        sep = "=" * 88
        lines = [
            f"\n{sep}",
            f"[Pure-OPRD-MT step={step:>6d}] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  mid_layers={indices['mid_layers']} last_layers={indices['last_layers']}",
            f"  middle_steps={indices['middle_steps']} late_steps={indices['late_steps']}",
            f"  weights: mid_il={self.mid_il_weight:.3f}, last_il={self.last_il_weight:.3f}, "
            f"last_rl={self.last_rl_weight:.3f}, final={self.final_repr_weight:.3f}",
            "-" * 88,
        ]
        for k in sorted(payload):
            lines.append(f"  {k:<36s}: {payload[k]:.8f}")
        lines.append(sep)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ──────────────────────────────────────────────────────────────────────
    # Main training entry point
    # ──────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        student_planner,
        teacher_il_planner,
        teacher_rl_planner,
        vl_features: torch.Tensor,
        action_input,
    ) -> BatchFeature:
        self._call_count += 1

        B = vl_features.shape[0]
        device = vl_features.device
        s_dtype = next(student_planner.parameters()).dtype

        his_traj = action_input.his_traj
        ego_status = action_input.status_feature

        # Phase 1: student on-policy chain, no gradient.
        with torch.no_grad():
            vl_s0, his_s0, ego_s0 = self._encode(student_planner, vl_features, his_traj, ego_status, s_dtype)
            chain = self._sample_chain(student_planner, vl_s0, his_s0, ego_s0, B, device, s_dtype)
        K = chain.shape[1] - 1
        ddim_t_list = [int(student_planner.ddim_t[i].item()) for i in range(K)]

        # Encode conditioning once per planner.  Teacher encoders are frozen.
        vl_s, his_s, ego_s = self._encode(student_planner, vl_features, his_traj, ego_status, s_dtype)
        with torch.no_grad():
            vl_il, his_il, ego_il = self._encode(teacher_il_planner, vl_features, his_traj, ego_status, torch.float32)
            vl_rl, his_rl, ego_rl = self._encode(teacher_rl_planner, vl_features, his_traj, ego_status, torch.float32)

        num_blocks = len(student_planner.model.transformer_blocks)
        indices = self._select_indices(num_blocks=num_blocks, num_steps=K)
        middle_steps = set(indices["middle_steps"])
        late_steps = set(indices["late_steps"])

        mid_il_losses: List[torch.Tensor] = []
        last_il_losses: List[torch.Tensor] = []
        last_rl_losses: List[torch.Tensor] = []
        final_il_losses: List[torch.Tensor] = []
        final_rl_losses: List[torch.Tensor] = []
        step_losses: List[torch.Tensor] = []

        for i in range(K):
            # Skip early steps by construction.  They are noisy and are not used
            # by the middle/late partition.
            if i not in middle_steps and i not in late_steps:
                continue

            z_t = chain[:, i].to(s_dtype)
            t_batch = student_planner.make_timesteps(B, ddim_t_list[i], device)
            idx_batch = student_planner.make_timesteps(B, i, device)

            _, _, _, out_s, hs_s = self._p_mean_variance_with_repr(
                student_planner, z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False
            )
            with torch.no_grad():
                _, _, _, out_il, hs_il = self._p_mean_variance_with_repr(
                    teacher_il_planner, z_t.float(), t_batch, idx_batch, vl_il, his_il, ego_il, deterministic=True
                )
                _, _, _, out_rl, hs_rl = self._p_mean_variance_with_repr(
                    teacher_rl_planner, z_t.float(), t_batch, idx_batch, vl_rl, his_rl, ego_rl, deterministic=True
                )

            current_terms: List[torch.Tensor] = []
            if i in middle_steps:
                mid_il = self._mean_layer_loss(hs_s, hs_il, indices["mid_layers"])
                mid_il_losses.append(mid_il.detach())
                current_terms.append(self.mid_il_weight * mid_il)

            if i in late_steps:
                last_il = self._mean_layer_loss(hs_s, hs_il, indices["last_layers"])
                last_rl = self._mean_layer_loss(hs_s, hs_rl, indices["last_layers"])
                last_il_losses.append(last_il.detach())
                last_rl_losses.append(last_rl.detach())
                current_terms.append(self.last_il_weight * last_il + self.last_rl_weight * last_rl)

                if self.use_final_repr and self.final_repr_weight > 0.0:
                    final_il = self._repr_distance(out_s, out_il)
                    final_rl = self._repr_distance(out_s, out_rl)
                    final_il_losses.append(final_il.detach())
                    final_rl_losses.append(final_rl.detach())
                    current_terms.append(
                        self.final_repr_weight
                        * (self.last_il_weight * final_il + self.last_rl_weight * final_rl)
                    )

            if current_terms:
                step_losses.append(torch.stack([term.to(out_s.dtype) for term in current_terms]).sum())

        if step_losses:
            oprd_loss = torch.stack(step_losses).mean()
        else:
            # Fallback should not happen unless K is invalid.
            oprd_loss = vl_features.new_zeros(())

        loss = oprd_loss
        if not torch.isfinite(loss):
            loss = loss.new_zeros(())

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if values:
                return torch.stack([v.float() for v in values]).mean().to(device)
            return torch.zeros((), device=device, dtype=torch.float32)

        mid_il_mean = _mean_or_zero(mid_il_losses)
        last_il_mean = _mean_or_zero(last_il_losses)
        last_rl_mean = _mean_or_zero(last_rl_losses)
        final_il_mean = _mean_or_zero(final_il_losses)
        final_rl_mean = _mean_or_zero(final_rl_losses)
        step_loss_tensor = torch.stack([v.detach().float() for v in step_losses]) if step_losses else torch.zeros(1, device=device)

        payload = {
            "oprd_loss": float(oprd_loss.detach().float().item()),
            "oprd_mid_il_loss": float(mid_il_mean.item()),
            "oprd_last_il_loss": float(last_il_mean.item()),
            "oprd_last_rl_loss": float(last_rl_mean.item()),
            "oprd_final_il_loss": float(final_il_mean.item()),
            "oprd_final_rl_loss": float(final_rl_mean.item()),
            "oprd_step_loss_mean": float(step_loss_tensor.mean().item()),
            "oprd_step_loss_max": float(step_loss_tensor.max().item()),
        }
        if self.log_dir is not None and self._call_count % self.log_interval == 0 and self._local_rank == 0:
            self._write_text_log(self._call_count, indices, payload)

        return BatchFeature(data={
            "loss": loss,
            "oprd_loss": oprd_loss.detach(),
            "distill_loss": oprd_loss.detach(),  # compatibility with existing Lightning logging
            "oprd_mid_il_loss": mid_il_mean.detach(),
            "oprd_last_il_loss": last_il_mean.detach(),
            "oprd_last_rl_loss": last_rl_mean.detach(),
            "oprd_final_il_loss": final_il_mean.detach(),
            "oprd_final_rl_loss": final_rl_mean.detach(),
            "oprd_step_loss_mean": step_loss_tensor.mean().detach(),
            "oprd_step_loss_max": step_loss_tensor.max().detach(),
            "oprd_mid_il_weight": torch.tensor(self.mid_il_weight, device=device),
            "oprd_last_il_weight": torch.tensor(self.last_il_weight, device=device),
            "oprd_last_rl_weight": torch.tensor(self.last_rl_weight, device=device),
            "oprd_final_repr_weight": torch.tensor(self.final_repr_weight, device=device),
            "oprd_middle_step_count": torch.tensor(float(len(indices["middle_steps"])), device=device),
            "oprd_late_step_count": torch.tensor(float(len(indices["late_steps"])), device=device),
            "oprd_mid_layer_count": torch.tensor(float(len(indices["mid_layers"])), device=device),
            "oprd_last_layer_count": torch.tensor(float(len(indices["last_layers"])), device=device),
            "denoising_steps": torch.tensor(float(K), device=device),
            "chain_abs_max": chain.float().abs().max().detach(),
        })

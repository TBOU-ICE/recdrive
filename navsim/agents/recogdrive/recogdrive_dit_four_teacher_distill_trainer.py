"""
Four-teacher DiT OPD distillation for ReCogDrive.

This is the fixed-weight extension of the dual-teacher DiT OPD trainer:
for each denoising step from the student's own stochastic DDIM chain, the
student is matched to four frozen teacher DiTs at the same (z_t, t).
"""

from typing import Dict, List

import torch
from transformers.feature_extraction_utils import BatchFeature


class ReCogDriveDiTFourTeacherDistillTrainer:
    """Fixed-weight OPD distillation from four frozen DiT teachers."""

    def __init__(
        self,
        teacher_weights: Dict[str, float],
        min_sigma: float = 0.04,
        smooth_weight: float = 0.02,
    ):
        if not teacher_weights:
            raise ValueError("teacher_weights must not be empty")
        total_weight = float(sum(teacher_weights.values()))
        if total_weight <= 0:
            raise ValueError("teacher weights must sum to a positive value")

        self.teacher_weights = {
            name: float(weight) / total_weight
            for name, weight in teacher_weights.items()
        }
        self.teacher_names: List[str] = list(self.teacher_weights.keys())
        self.min_sigma = float(min_sigma)
        self.smooth_weight = float(smooth_weight)

    @staticmethod
    def _encode(planner, vl_features, his_traj, ego_status, dtype):
        """Encode conditioning tensors through a planner's projection layers."""
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
        """Sample a detached stochastic DDIM chain from the current student."""
        horizon = planner.config.action_horizon
        action_dim = planner.config.action_dim
        z = torch.randn((batch_size, horizon, action_dim), device=device, dtype=dtype)
        chain = [z.clone()]

        for i in range(planner.ddim_steps):
            t_batch = planner.make_timesteps(batch_size, int(planner.ddim_t[i].item()), device)
            idx_batch = planner.make_timesteps(batch_size, i, device)
            mu, logvar, _ = planner.p_mean_variance(
                z,
                t_batch,
                idx_batch,
                vl_embeds,
                his_embeds,
                ego_embeds,
                deterministic=False,
            )
            sigma = self._safe_sigma(logvar, self.min_sigma, dtype)
            noise = torch.randn_like(z).clamp_(-5.0, 5.0)
            z = (mu + sigma * noise).detach()
            chain.append(z.clone())

        return torch.stack(chain, dim=1)

    def compute_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        vl_features: torch.Tensor,
        action_input,
    ) -> BatchFeature:
        """Compute fixed 0.25/0.25/0.25/0.25 four-teacher OPD loss."""
        if set(teacher_planners.keys()) != set(self.teacher_names):
            raise ValueError(
                f"teacher_planners keys {sorted(teacher_planners.keys())} do not match "
                f"configured weights {sorted(self.teacher_names)}"
            )

        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature

        with torch.no_grad():
            vl_s0, his_s0, ego_s0 = self._encode(
                student_planner, vl_features, his_traj, ego_status, student_dtype
            )
            chain = self._sample_chain(
                student_planner,
                vl_s0,
                his_s0,
                ego_s0,
                batch_size,
                device,
                student_dtype,
            )

        num_steps = chain.shape[1] - 1
        ddim_t_list = [int(student_planner.ddim_t[i].item()) for i in range(num_steps)]

        teacher_encodings = {}
        with torch.no_grad():
            for name, teacher in teacher_planners.items():
                teacher.eval()
                teacher_encodings[name] = self._encode(
                    teacher, vl_features, his_traj, ego_status, torch.float32
                )

        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, student_dtype
        )

        total_loss = vl_features.new_zeros(())
        per_teacher_step_losses = {
            name: [] for name in self.teacher_names
        }
        weighted_step_losses = []
        raw_mix_step_losses = []
        sigma_list = []
        mu_s_last = None
        teacher_mix_last = None

        for i in range(num_steps):
            z_t = chain[:, i].to(student_dtype)
            t_batch = student_planner.make_timesteps(batch_size, ddim_t_list[i], device)
            idx_batch = student_planner.make_timesteps(batch_size, i, device)

            mu_s, logvar_s, _ = student_planner.p_mean_variance(
                z_t,
                t_batch,
                idx_batch,
                vl_s,
                his_s,
                ego_s,
                deterministic=False,
            )
            sigma_s = self._safe_sigma(logvar_s, self.min_sigma, student_dtype).detach()
            sigma2 = sigma_s.float().pow(2).clamp(min=1e-6)

            step_loss = vl_features.new_zeros(())
            teacher_mix = torch.zeros_like(mu_s.float())
            for name in self.teacher_names:
                teacher = teacher_planners[name]
                vl_t, his_t, ego_t = teacher_encodings[name]
                with torch.no_grad():
                    mu_t, _, _ = teacher.p_mean_variance(
                        z_t.float(),
                        t_batch,
                        idx_batch,
                        vl_t,
                        his_t,
                        ego_t,
                        deterministic=True,
                    )

                diff = mu_s.float() - mu_t.detach()
                kl_per_sample = diff.pow(2).div(2.0 * sigma2).sum(dim=(1, 2))
                kl_mean = kl_per_sample.mean()
                per_teacher_step_losses[name].append(kl_mean.detach())
                step_loss = step_loss + self.teacher_weights[name] * kl_mean.to(step_loss.dtype)
                teacher_mix = teacher_mix + self.teacher_weights[name] * mu_t.detach()

            total_loss = total_loss + step_loss
            weighted_step_losses.append(step_loss.detach())
            sigma_list.append(sigma_s.detach().float().mean())

            diff_mix = mu_s.float() - teacher_mix
            horizon = diff_mix.shape[1]
            action_dim = diff_mix.shape[2]
            raw_mix_step_losses.append(
                (diff_mix.pow(2).sum(dim=(1, 2)) / (horizon * action_dim)).detach().mean()
            )

            if i == num_steps - 1:
                mu_s_last = mu_s
                teacher_mix_last = teacher_mix.detach()

        distill_loss = total_loss / num_steps
        pred_traj_s_for_loss = student_planner.denorm_odo(mu_s_last.float())
        smooth_loss = self._jerk_loss(pred_traj_s_for_loss)
        loss = distill_loss + self.smooth_weight * smooth_loss

        if not torch.isfinite(loss):
            loss = loss.new_zeros(())

        step_kls = torch.stack(weighted_step_losses)
        sigma_mean = torch.stack(sigma_list).mean()
        with torch.no_grad():
            pred_traj_s = pred_traj_s_for_loss.detach()
            pred_traj_teacher_mix = student_planner.denorm_odo(teacher_mix_last.float()).to(device)
            pred_traj_l1_to_teacher_mix = torch.nn.functional.l1_loss(
                pred_traj_s, pred_traj_teacher_mix
            )

        data = {
            "loss": loss,
            "distill_loss": distill_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
            "weighted_smooth_loss": (self.smooth_weight * smooth_loss).detach(),
            "transition_kl": step_kls.mean().detach(),
            "step_kl_mean": step_kls.mean().detach(),
            "step_kl_max": step_kls.max().detach(),
            "step_kls": step_kls.detach(),
            "sigma_mean": sigma_mean.detach(),
            "chain_abs_max": chain.float().abs().max().detach(),
            "denoising_steps": torch.tensor(float(num_steps), device=device),
            "pred_traj_l1_to_teacher_mix": pred_traj_l1_to_teacher_mix.detach(),
            "student_pred_traj_mean": pred_traj_s.mean().detach(),
            "student_pred_traj_std": pred_traj_s.std(unbiased=False).detach(),
            "teacher_mix_pred_traj_mean": pred_traj_teacher_mix.mean().detach(),
            "teacher_mix_pred_traj_std": pred_traj_teacher_mix.std(unbiased=False).detach(),
            "teacher_student_pred_traj_abs_mean": (
                pred_traj_s - pred_traj_teacher_mix
            ).abs().mean().detach(),
            "student_pred_traj_first_point": pred_traj_s[0, 0].detach(),
            "teacher_mix_pred_traj_first_point": pred_traj_teacher_mix[0, 0].detach(),
        }

        for name in self.teacher_names:
            values = torch.stack(per_teacher_step_losses[name])
            data[f"kl_{name}_mean"] = values.mean().detach()
            data[f"teacher_weight_{name}"] = torch.tensor(self.teacher_weights[name], device=device)

        for i, value in enumerate(weighted_step_losses):
            data[f"kl_step_{i}"] = value.detach()
        for i, value in enumerate(raw_mix_step_losses):
            data[f"kl_raw_mix_step_{i}"] = value.detach()
        for i, value in enumerate(sigma_list):
            data[f"sigma_step_{i}"] = value.detach()

        return BatchFeature(data=data)

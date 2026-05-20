"""Diffusion-transition OPD trainer for ReCogDrive DiT action planners.

This trainer implements the clean setting discussed for ReCogDrive-OPD:

  fixed/cached VLM hidden state h
  frozen RL-trained DiT teacher p_phi(z_{t-1} | z_t, h)
  trainable IL-initialized DiT student p_theta(z_{t-1} | z_t, h)

The student first samples its own denoising chain.  The teacher is then queried on
exactly the same student on-policy noisy states z_t.  The training loss is the
closed-form reverse-transition KL KL(p_theta || p_phi).  With the shared
DDPM/DDIM variance schedule used by ReCogDrive this reduces to a variance-
weighted mean matching loss between the reverse-process means.

This is intentionally different from the older `recogdrive_dit_distill_trainer`:
it does not use token top-k OPD, feature matching, final trajectory matching, or
PDM reward weighting.  It is a pure continuous diffusion-transition OPD loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.feature_extraction_utils import BatchFeature


@dataclass
class DiTOPDStats:
    loss: torch.Tensor
    transition_kl: torch.Tensor
    pred_traj_l1_to_teacher: torch.Tensor
    denoising_steps: torch.Tensor


def _repeat_batch_feature(action_input: BatchFeature, repeats: int) -> BatchFeature:
    """Repeat every tensor in a BatchFeature by `repeats` along batch dimension."""
    if repeats <= 1:
        return action_input
    data = {}
    for key, value in action_input.items():
        if torch.is_tensor(value):
            data[key] = value.repeat_interleave(repeats, dim=0)
        else:
            data[key] = value
    return BatchFeature(data=data)


def _build_common_embeddings(planner, vl_features: torch.Tensor, action_input: BatchFeature):
    """Build planner-specific conditioning embeddings from the same VLM hidden state."""
    vl_embeds = planner.feature_encoder(vl_features)
    his_traj_features = planner.his_traj_encoder(
        action_input.his_traj.unsqueeze(1)
    ).repeat(1, planner.config.action_horizon, 1)
    ego_status_features = planner.ego_status_encoder(action_input.status_feature)
    return vl_embeds, his_traj_features, ego_status_features


def _ensure_sampling_defaults(planner) -> None:
    """Make stochastic denoising usable outside the original GRPO path.

    ReCogDrive defines attributes such as ``min_sampling_denoising_std`` in
    ``_init_grpo``.  DiT-OPD intentionally does not enable GRPO, but it still
    needs stochastic student denoising chains.  We therefore install the same
    safe defaults from ``planner.config.grpo_cfg`` when those runtime attributes
    are absent.  This keeps OPD independent from the PDM reward/GRPO machinery.
    """
    cfg = getattr(getattr(planner, "config", None), "grpo_cfg", None)
    defaults = {
        "denoised_clip_value": 1.0,
        "eval_randn_clip_value": 1.0,
        "randn_clip_value": 5.0,
        "final_action_clip_value": 1.0,
        "eps_clip_value": None,
        "eval_min_sampling_denoising_std": 1e-4,
        "min_sampling_denoising_std": 0.04,
        "min_logprob_denoising_std": 0.1,
    }
    for name, fallback in defaults.items():
        if not hasattr(planner, name):
            setattr(planner, name, getattr(cfg, name, fallback) if cfg is not None else fallback)


def _transition_kl_on_student_chain(
    student_planner,
    teacher_planner,
    vl_features: torch.Tensor,
    action_input: BatchFeature,
    chains: torch.Tensor,
    min_kl_variance: float,
    max_kl_per_step: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute closed-form reverse-transition KL on a student denoising chain.

    Args:
        chains: Tensor with shape (B, K + 1, H, D), where chains[:, i] is z_t
            and chains[:, i + 1] is z_{t-1} for denoising step i.

    Returns:
        transition_kl: scalar mean KL over denoising steps and batch.
        step_kls: tensor of shape (K,) with detached mean KL per denoising step.
    """
    if student_planner.config.sampling_method not in ("ddpm", "ddim"):
        raise ValueError(
            "DiT-OPD currently supports DDPM/DDIM action heads only; "
            f"got sampling_method={student_planner.config.sampling_method!r}."
        )
    if student_planner.config.sampling_method != teacher_planner.config.sampling_method:
        raise ValueError(
            "Student and teacher must use the same sampling_method for transition KL: "
            f"student={student_planner.config.sampling_method}, "
            f"teacher={teacher_planner.config.sampling_method}."
        )

    B, K1, H, D = chains.shape
    num_steps = K1 - 1
    device = chains.device

    vl_s, his_s, ego_s = _build_common_embeddings(student_planner, vl_features, action_input)
    with torch.no_grad():
        vl_t, his_t, ego_t = _build_common_embeddings(teacher_planner, vl_features, action_input)

    if student_planner.config.sampling_method == "ddpm":
        step_size = student_planner.config.ddpm_cfg.num_train_timesteps // student_planner.config.num_inference_steps
        timesteps = list(reversed(range(0, student_planner.config.ddpm_cfg.num_train_timesteps, step_size)))
    else:
        timesteps = [int(t.item()) for t in student_planner.ddim_t]

    if len(timesteps) != num_steps:
        raise RuntimeError(
            f"Denoising chain has {num_steps} transitions, but scheduler produced {len(timesteps)} timesteps."
        )

    step_losses = []
    for i, t_int in enumerate(timesteps):
        # Student on-policy state z_t.  The chain is detached; gradients should
        # update the transition density at z_t, not backprop through sampling.
        x_t = chains[:, i].detach()
        t_batch = student_planner.make_timesteps(B, t_int, device)
        if student_planner.config.sampling_method == "ddim":
            index_batch = student_planner.make_timesteps(B, i, device)
        else:
            index_batch = t_batch

        mean_s, logvar_s, _ = student_planner.p_mean_variance(
            x_t,
            t_batch,
            index_batch,
            vl_s,
            his_s,
            ego_s,
            deterministic=False,
        )
        with torch.no_grad():
            mean_t, logvar_t, _ = teacher_planner.p_mean_variance(
                x_t,
                t_batch,
                index_batch,
                vl_t,
                his_t,
                ego_t,
                deterministic=False,
            )

        # Closed-form reverse KL between Gaussian reverse transitions:
        # KL(N(mu_s, var_s) || N(mu_t, var_t)).  When the scheduler variance is
        # shared, this exactly reduces to 0.5 * ||mu_s - mu_t||^2 / var_t.
        # Keeping the full formula makes the implementation robust if DDIM eta
        # or another variance parameter differs across checkpoints.
        var_s = torch.exp(logvar_s.float()).clamp(min=min_kl_variance)
        var_t = torch.exp(logvar_t.float()).clamp(min=min_kl_variance)
        mean_delta_sq = (mean_s.float() - mean_t.float()).pow(2)
        kl_elem = 0.5 * (torch.log(var_t) - torch.log(var_s) + (var_s + mean_delta_sq) / var_t - 1.0)
        kl_elem = torch.nan_to_num(kl_elem, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        kl_step = kl_elem.mean()
        if max_kl_per_step is not None and max_kl_per_step > 0:
            kl_step = kl_step.clamp(max=float(max_kl_per_step))
        step_losses.append(kl_step)

    step_kls = torch.stack(step_losses)
    transition_kl = step_kls.mean()
    transition_kl = torch.nan_to_num(transition_kl, nan=0.0, posinf=0.0, neginf=0.0)
    return transition_kl, step_kls.detach()


class ReCogDriveDiTOPDTrainer:
    """Pure DiT diffusion-transition OPD for a frozen teacher planner."""

    def __init__(
        self,
        sample_time: int = 1,
        min_kl_variance: float = 1e-4,
        max_kl_per_step: Optional[float] = None,
        rollout_deterministic: bool = False,
        teacher_eval_deterministic: bool = True,
    ):
        self.sample_time = int(sample_time)
        if self.sample_time < 1:
            raise ValueError("sample_time must be >= 1")
        self.min_kl_variance = float(min_kl_variance)
        self.max_kl_per_step = None if max_kl_per_step is None else float(max_kl_per_step)
        self.rollout_deterministic = bool(rollout_deterministic)
        self.teacher_eval_deterministic = bool(teacher_eval_deterministic)

    def compute_loss(
        self,
        student_planner,
        teacher_planner,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
    ) -> BatchFeature:
        """Return OPD loss and logging tensors.

        The same `vl_features` tensor is fed into both student and teacher action
        planners.  If `sample_time > 1`, each condition is repeated so the student
        draws multiple on-policy denoising chains per NAVSIM scene.
        """
        teacher_planner.eval()
        if hasattr(student_planner, "set_frozen_modules_to_eval_mode"):
            student_planner.set_frozen_modules_to_eval_mode()

        if self.sample_time > 1:
            vl_features_rep = vl_features.repeat_interleave(self.sample_time, dim=0)
            action_input_rep = _repeat_batch_feature(action_input, self.sample_time)
        else:
            vl_features_rep = vl_features
            action_input_rep = action_input

        # On-policy student chain.  The chain is detached inside sample_chain;
        # the KL step below recomputes student transitions with gradients.
        _ensure_sampling_defaults(student_planner)
        _ensure_sampling_defaults(teacher_planner)
        with torch.no_grad():
            chains, pred_traj = student_planner.sample_chain(
                vl_features_rep,
                action_input_rep.his_traj,
                action_input_rep.status_feature,
                deterministic=self.rollout_deterministic,
            )

        transition_kl, step_kls = _transition_kl_on_student_chain(
            student_planner=student_planner,
            teacher_planner=teacher_planner,
            vl_features=vl_features_rep,
            action_input=action_input_rep,
            chains=chains,
            min_kl_variance=self.min_kl_variance,
            max_kl_per_step=self.max_kl_per_step,
        )

        with torch.no_grad():
            teacher_pred_traj = teacher_planner.get_action(
                vl_features_rep,
                action_input_rep,
                deterministic=self.teacher_eval_deterministic,
            )["pred_traj"]
            pred_traj_f = pred_traj.float()
            teacher_pred_traj_f = teacher_pred_traj.float()
            traj_l1 = torch.nn.functional.l1_loss(pred_traj_f, teacher_pred_traj_f)
            traj_abs_mean = (pred_traj_f - teacher_pred_traj_f).abs().mean()

        loss = torch.nan_to_num(transition_kl, nan=0.0, posinf=0.0, neginf=0.0)
        return BatchFeature(data={
            "loss": loss,
            "opd_loss": loss.detach(),
            "transition_kl": transition_kl.detach(),
            "step_kl_mean": step_kls.mean().detach(),
            "step_kl_max": step_kls.max().detach(),
            "step_kls": step_kls.detach(),
            "pred_traj_l1_to_teacher": traj_l1.detach(),
            "denoising_steps": torch.tensor(float(chains.shape[1] - 1), device=loss.device),
            "student_pred_traj_mean": pred_traj_f.mean().detach(),
            "student_pred_traj_std": pred_traj_f.std(unbiased=False).detach(),
            "teacher_pred_traj_mean": teacher_pred_traj_f.mean().detach(),
            "teacher_pred_traj_std": teacher_pred_traj_f.std(unbiased=False).detach(),
            "teacher_student_pred_traj_abs_mean": traj_abs_mean.detach(),
            "student_pred_traj_first_point": pred_traj_f[0, 0].detach(),
            "teacher_pred_traj_first_point": teacher_pred_traj_f[0, 0].detach(),
        })

"""Routed privileged OPD v2 trainer.

Core invariants:
- rollout/history is sampled ONLY from the goal-free student;
- routed teacher receives the exact same student z_t plus privileged GT goal;
- teacher and student reverse means are reconstructed with the same student
  DDIM sigma before the Gaussian KL surrogate is formed;
- no predicted goal is fed into the deployment student;
- optional B/C/F/G losses are additive and disabled by default.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_dit_scene_router_distill_trainer import (
    ReCogDriveDiTSceneRouterDistillTrainer,
)
from .goal_adapter_planner import extract_goal_points

_LOG_2PI_E = math.log(2.0 * math.pi * math.e)


class PrivilegedOPDV2Trainer(ReCogDriveDiTSceneRouterDistillTrainer):
    def __init__(
        self,
        *args,
        kd_weight: float = 1.0,
        task_weight: float = 0.10,
        goal_aux_weight: float = 0.0,
        goal_pref_weight: float = 0.0,
        goal_pref_temperature_m: float = 2.0,
        goal_pref_candidate_count: int = 8,
        goal_pref_geo_weight: float = 0.25,
        residual_privilege: bool = False,
        precision_clip: float = 25.0,
        **kwargs,
    ):
        # x0/mu knobs in the old trainer are intentionally ignored here; v2
        # always constructs shared-sigma reverse means explicitly.
        kwargs["match_target"] = "mu"
        kwargs["exopd_lambda"] = 1.0
        super().__init__(*args, **kwargs)
        self.kd_weight = float(kd_weight)
        self.task_weight = float(task_weight)
        self.goal_aux_weight = float(goal_aux_weight)
        self.goal_pref_weight = float(goal_pref_weight)
        self.goal_pref_temperature_m = float(goal_pref_temperature_m)
        self.goal_pref_candidate_count = int(goal_pref_candidate_count)
        self.goal_pref_geo_weight = float(goal_pref_geo_weight)
        if self.goal_pref_candidate_count <= 0:
            raise ValueError("goal_pref_candidate_count must be > 0")
        if self.goal_pref_geo_weight < 0:
            raise ValueError("goal_pref_geo_weight must be >= 0")
        self.residual_privilege = bool(residual_privilege)
        self.precision_clip = float(precision_clip)
        if self.precision_clip <= 0:
            raise ValueError("precision_clip must be > 0")

    @staticmethod
    def _shared_ddim_mean(planner, z_t, idx_batch, x0, sigma):
        z = z_t.float()
        x0 = x0.float()
        sigma = sigma.float()
        alpha_t = planner.extract(planner.ddim_alphas, idx_batch, z.shape).float()
        alpha_prev = planner.extract(planner.ddim_alphas_prev, idx_batch, z.shape).float()
        sqrt_oma = planner.extract(planner.ddim_sqrt_one_minus_alphas, idx_batch, z.shape).float().clamp(min=1e-8)
        pred_noise = (z - alpha_t.sqrt() * x0) / sqrt_oma
        pred_dir = (1.0 - alpha_prev - sigma.pow(2)).clamp(min=0).sqrt() * pred_noise
        return alpha_prev.sqrt() * x0 + pred_dir

    def _raw_precision(self, sigma_geom: torch.Tensor) -> torch.Tensor:
        sigma2 = sigma_geom.float().pow(2).clamp(min=self.min_sigma ** 2, max=1e6)
        return (0.5 / sigma2).clamp(max=self.precision_clip)

    @staticmethod
    def _waypoint_gap_m(planner, x0_a: torch.Tensor, x0_b: torch.Tensor) -> torch.Tensor:
        """Mean per-waypoint xy gap in metres between normalized trajectories."""
        a = planner.denorm_odo(x0_a.float())
        b = planner.denorm_odo(x0_b.float())
        return (a[..., :2] - b[..., :2]).norm(dim=-1).mean()

    @staticmethod
    def _waypoint_gap_m_per_sample(planner, x0_a: torch.Tensor, x0_b: torch.Tensor) -> torch.Tensor:
        """Per-sample mean waypoint xy gap in metres."""
        a = planner.denorm_odo(x0_a.float())
        b = planner.denorm_odo(x0_b.float())
        return (a[..., :2] - b[..., :2]).norm(dim=-1).mean(dim=-1)

    def compute_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        ref_planner: Optional[torch.nn.Module],
        vl_features: torch.Tensor,
        action_input,
        bucket_per_sample: List[str],
        aux_goal_head=None,
        goal_preference_head=None,
        candidate_goals_raw: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student_planner.parameters()).dtype
        his_traj = action_input.his_traj
        ego_status = action_input.status_feature
        gt_traj = action_input.action.float()

        # Student encodings with grad. These same observable encodings feed the
        # optional B/C heads, but the planner itself remains goal-free.
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, student_dtype
        )

        # Detached on-policy student DDIM chain.
        with torch.no_grad():
            chain = self._sample_chain(
                student_planner,
                vl_s.detach(), his_s.detach(), ego_s.detach(),
                batch_size, device, student_dtype,
            )
        num_steps = chain.shape[1] - 1
        ddim_t_list = [int(student_planner.ddim_t[i].item()) for i in range(num_steps)]

        # Frozen encodings.
        teacher_encodings = {}
        with torch.no_grad():
            for name, teacher in teacher_planners.items():
                teacher.eval()
                teacher_encodings[name] = self._encode(
                    teacher, vl_features, his_traj, ego_status, torch.float32
                )
            ref_encoding = None
            if self.residual_privilege:
                if ref_planner is None:
                    raise RuntimeError("residual_privilege=True requires a frozen ref_planner")
                ref_encoding = self._encode(
                    ref_planner, vl_features, his_traj, ego_status, torch.float32
                )

        resolved = [self._resolve_bucket(b) for b in bucket_per_sample]
        bucket_to_indices: Dict[str, List[int]] = {}
        for i, bucket in enumerate(resolved):
            bucket_to_indices.setdefault(bucket, []).append(i)

        # Accumulate raw precision-weighted losses; normalize once across all
        # timesteps so relative timestep precision survives while E[w] ~= 1.
        raw_kd_sum = vl_features.new_zeros((), dtype=torch.float32)
        precision_means = []
        n_total = 0
        x0_gap_steps = []
        sigma_steps = []
        last_student_x0 = None
        last_teacher_x0 = torch.zeros_like(chain[:, 0], dtype=torch.float32)
        last_teacher_mu = torch.zeros_like(chain[:, 0], dtype=torch.float32)
        final_z_t = final_t_batch = final_idx_batch = final_sigma_geom = None

        for step in range(num_steps):
            z_t = chain[:, step].to(student_dtype)
            t_batch = student_planner.make_timesteps(batch_size, ddim_t_list[step], device)
            idx_batch = student_planner.make_timesteps(batch_size, step, device)

            _, logvar_s, x0_s = student_planner.p_mean_variance(
                z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False
            )
            sigma_geom = (0.5 * logvar_s.float().clamp(-40, 20)).exp().detach()
            precision = self._raw_precision(sigma_geom)
            precision_means.append(precision.mean().detach())
            sigma_steps.append(sigma_geom.mean().detach())
            mu_s = self._shared_ddim_mean(student_planner, z_t, idx_batch, x0_s, sigma_geom)

            step_gap = vl_features.new_zeros((), dtype=torch.float32)
            for bucket, indices in bucket_to_indices.items():
                sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                teacher = teacher_planners[bucket]
                enc = teacher_encodings[bucket]
                # Each teacher decides whether this is final-point or multi3.
                goals = extract_goal_points(
                    gt_traj[sel], teacher.goal_point_mode, teacher.goal_indices
                )
                with torch.no_grad():
                    with teacher.goal_context(goals):
                        _, _, x0_t_on = teacher.p_mean_variance(
                            z_t[sel].float(), t_batch[sel], idx_batch[sel],
                            enc[0][sel], enc[1][sel], enc[2][sel], deterministic=True,
                        )
                    mu_t_on = self._shared_ddim_mean(
                        student_planner, z_t[sel], idx_batch[sel], x0_t_on, sigma_geom[sel]
                    )

                    if self.residual_privilege:
                        # Isolate the privileged residual: teacher(on)-teacher(off),
                        # then add it to a frozen student-family reference policy.
                        _, _, x0_t_off = teacher.p_mean_variance(
                            z_t[sel].float(), t_batch[sel], idx_batch[sel],
                            enc[0][sel], enc[1][sel], enc[2][sel], deterministic=True,
                        )
                        mu_t_off = self._shared_ddim_mean(
                            student_planner, z_t[sel], idx_batch[sel], x0_t_off, sigma_geom[sel]
                        )
                        _, _, x0_ref = ref_planner.p_mean_variance(
                            z_t[sel].float(), t_batch[sel], idx_batch[sel],
                            ref_encoding[0][sel], ref_encoding[1][sel], ref_encoding[2][sel],
                            deterministic=True,
                        )
                        mu_ref = self._shared_ddim_mean(
                            student_planner, z_t[sel], idx_batch[sel], x0_ref, sigma_geom[sel]
                        )
                        target_mu = mu_ref + (mu_t_on - mu_t_off)
                    else:
                        target_mu = mu_t_on

                    if step == num_steps - 1:
                        last_teacher_x0[sel] = x0_t_on.float()
                        last_teacher_mu[sel] = mu_t_on.float()

                diff2 = (mu_s[sel].float() - target_mu.detach()).pow(2).mean(dim=(1, 2))
                w = precision[sel].mean(dim=(1, 2))
                raw_kd_sum = raw_kd_sum + (diff2 * w).sum()
                n_total += len(indices)
                step_gap = step_gap + self._waypoint_gap_m(
                    student_planner, x0_s[sel], x0_t_on
                ) * len(indices)
            x0_gap_steps.append((step_gap / max(batch_size, 1)).detach())
            if step == num_steps - 1:
                last_student_x0 = x0_s
                final_z_t = z_t.detach().float()
                final_t_batch = t_batch.detach()
                final_idx_batch = idx_batch.detach()
                final_sigma_geom = sigma_geom.detach().float()

        precision_norm = torch.stack(precision_means).mean().clamp(min=1e-6)
        # n_total = B * num_steps because routing uses exactly one teacher/sample.
        distill_loss = raw_kd_sum / max(n_total, 1) / precision_norm

        # Small task-preservation loss on the real GT trajectory. This is ordinary
        # diffusion IL, not another RL pass.
        task_loss = vl_features.new_zeros((), dtype=torch.float32)
        if self.task_weight > 0:
            task_loss = student_planner(vl_features, action_input).loss.float()

        # B: training-only endpoint internalization. Prediction is never fed to planner.
        goal_aux_loss = vl_features.new_zeros((), dtype=torch.float32)
        goal_aux_fde_m = vl_features.new_zeros((), dtype=torch.float32)
        if self.goal_aux_weight > 0:
            if aux_goal_head is None:
                raise RuntimeError("goal_aux_weight > 0 but aux_goal_head is missing")
            pred_goal_norm = aux_goal_head(vl_s, his_s, ego_s)
            gt_goal = gt_traj[:, -1:, :]
            gt_goal_norm = student_planner.norm_odo(gt_goal).squeeze(1).to(pred_goal_norm.dtype)
            goal_aux_loss = F.smooth_l1_loss(pred_goal_norm, gt_goal_norm[..., :pred_goal_norm.shape[-1]])
            # Metric in metres; fill heading=0 when head predicts xy only.
            padded = torch.zeros(batch_size, 1, 3, device=device, dtype=pred_goal_norm.dtype)
            padded[:, 0, :pred_goal_norm.shape[-1]] = pred_goal_norm
            pred_goal_raw = student_planner.denorm_odo(padded)[:, 0]
            goal_aux_fde_m = (pred_goal_raw[:, :2].float() - gt_traj[:, -1, :2]).norm(dim=-1).mean().detach()

        # C: privileged Goal Preference Distillation.  The student scores the
        # whole offline vocabulary, while the routed privileged teacher ranks a
        # small nearest-candidate shortlist by *policy consistency*: candidate
        # goal -> teacher trajectory is compared against the same teacher under
        # the true privileged goal. This is stronger than merely regressing the
        # GT endpoint and keeps teacher/student deployment conditioning separate.
        goal_pref_loss = vl_features.new_zeros((), dtype=torch.float32)
        goal_pref_top1_fde_m = vl_features.new_zeros((), dtype=torch.float32)
        goal_pref_teacher_gap_m = vl_features.new_zeros((), dtype=torch.float32)
        if self.goal_pref_weight > 0:
            if goal_preference_head is None or candidate_goals_raw is None:
                raise RuntimeError("goal_pref_weight > 0 requires preference head + candidate vocabulary")
            if any(teacher_planners[b].goal_point_mode != "final" for b in bucket_to_indices):
                raise RuntimeError("Goal Preference Distillation v2-C currently requires final-point teachers; use D separately for multi3 ablations.")
            if final_z_t is None:
                raise RuntimeError("missing final student rollout state for goal preference ranking")

            vocab = candidate_goals_raw.to(device=device, dtype=torch.float32)
            vocab_norm = student_planner.norm_odo(vocab.unsqueeze(0)).squeeze(0)
            logits = goal_preference_head(vl_s, his_s, ego_s, vocab_norm).float()
            teacher_traj_raw = student_planner.denorm_odo(last_teacher_x0.float())
            teacher_endpoint = teacher_traj_raw[:, -1, :2]
            vocab_dist = torch.cdist(teacher_endpoint.float(), vocab[:, :2].float(), p=2)
            m = min(self.goal_pref_candidate_count, vocab.shape[0])
            shortlist = torch.topk(vocab_dist, k=m, dim=-1, largest=False).indices  # (B,M)
            q_full = torch.zeros_like(logits)
            teacher_gap_acc = []

            # Rank the shortlist with exactly one routed teacher per sample.
            for bucket, indices in bucket_to_indices.items():
                sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                teacher = teacher_planners[bucket]
                enc = teacher_encodings[bucket]
                short_idx = shortlist[sel]                       # (Bb,M)
                cand = vocab[short_idx]                           # (Bb,M,3)
                bb = sel.numel()
                goal_flat = cand.reshape(bb * m, 1, 3)
                z_rep = final_z_t[sel].repeat_interleave(m, dim=0)
                t_rep = final_t_batch[sel].repeat_interleave(m, dim=0)
                i_rep = final_idx_batch[sel].repeat_interleave(m, dim=0)
                sig_rep = final_sigma_geom[sel].repeat_interleave(m, dim=0)
                enc_rep = tuple(x[sel].repeat_interleave(m, dim=0) for x in enc)
                with torch.no_grad():
                    with teacher.goal_context(goal_flat):
                        _, _, x0_cand = teacher.p_mean_variance(
                            z_rep, t_rep, i_rep, enc_rep[0], enc_rep[1], enc_rep[2], deterministic=True
                        )
                    x0_priv = last_teacher_x0[sel].repeat_interleave(m, dim=0)
                    traj_gap = self._waypoint_gap_m_per_sample(
                        student_planner, x0_cand.float(), x0_priv.float()
                    ).reshape(bb, m)
                    geo_gap = vocab_dist[sel].gather(1, short_idx)
                    tau = max(self.goal_pref_temperature_m, 1e-3)
                    score = -(traj_gap + self.goal_pref_geo_weight * geo_gap) / tau
                    q_short = torch.softmax(score, dim=-1)
                rows = sel[:, None].expand(-1, m)
                q_full[rows, short_idx] = q_short
                teacher_gap_acc.append((traj_gap * q_short).sum(dim=-1))

            # KL(q_teacher || p_student). q is detached; p is normalized across
            # the full vocabulary, so probability mass outside the shortlist is
            # still penalized through the softmax normalizer.
            q_full = q_full.detach()
            logp = torch.log_softmax(logits, dim=-1)
            goal_pref_loss = F.kl_div(logp, q_full, reduction="batchmean")
            top1 = vocab[logits.argmax(dim=-1), :2]
            goal_pref_top1_fde_m = (top1 - teacher_endpoint).norm(dim=-1).mean().detach()
            if teacher_gap_acc:
                goal_pref_teacher_gap_m = torch.cat(teacher_gap_acc).mean().detach()

        pred_traj = student_planner.denorm_odo(last_student_x0.float())
        smooth_loss = self._jerk_loss(pred_traj)
        loss = (
            self.kd_weight * distill_loss
            + self.task_weight * task_loss
            + self.goal_aux_weight * goal_aux_loss
            + self.goal_pref_weight * goal_pref_loss
            + self.smooth_weight * smooth_loss
        )
        if not torch.isfinite(loss):
            loss = self._finite_anchor_loss(student_planner)

        return BatchFeature(data={
            "loss": loss,
            "opd_loss": distill_loss.detach(),
            "task_loss": task_loss.detach(),
            "goal_aux_loss": goal_aux_loss.detach(),
            "goal_pref_loss": goal_pref_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
            "weighted_opd_loss": (self.kd_weight * distill_loss).detach(),
            "weighted_task_loss": (self.task_weight * task_loss).detach(),
            "weighted_goal_aux_loss": (self.goal_aux_weight * goal_aux_loss).detach(),
            "weighted_goal_pref_loss": (self.goal_pref_weight * goal_pref_loss).detach(),
            "precision_norm": precision_norm.detach(),
            "sigma_mean": torch.stack(sigma_steps).mean().detach(),
            "x0_gap_m": torch.stack(x0_gap_steps).mean().detach(),
            "goal_aux_fde_m": goal_aux_fde_m,
            "goal_pref_top1_fde_m": goal_pref_top1_fde_m,
            "goal_pref_teacher_gap_m": goal_pref_teacher_gap_m,
            "denoising_steps": torch.tensor(float(num_steps), device=device),
            "residual_privilege": torch.tensor(float(self.residual_privilege), device=device),
            # Non-detached components are intentionally private: the custom
            # Lightning wrapper uses them every N steps to measure gradient
            # contribution without modifying .grad. They are never logged/saved.
            "_opd_component": self.kd_weight * distill_loss,
            "_task_component": self.task_weight * task_loss,
        })

"""
Research variants of the scene-router GOAL-teacher OPD distillation trainer.

Additive file: does NOT modify any existing implementation.  Subclasses
``ReCogDriveDiTSceneRouterGoalDistillTrainer`` and adds exactly one A/B knob,
``variant``, so the three ideas can be compared against the untouched baseline
(``recogdrive_dit_scene_router_goal_distill_trainer.py``) one variable at a time.

variant == 'kl'  (方案一: logprob-KL)
    Match the DDIM reverse-transition MEAN ``mu`` instead of the clean sample
    ``x0``.  With the precision weight ``1/(2 sigma_s^2)`` already applied, this
    is exactly the shared-variance Gaussian reverse-KL between the student and
    the (goal-conditioned) teacher transition kernels -- the driving analogue of
    OPD's per-token log-prob KL.  Implemented by forcing ``match_target='mu'``;
    the loss path is otherwise identical to the baseline.

variant == 'anchor'  (方案二: Manifold Anchor 正则)
    Keep the on-policy OPD loss and ADD a feasibility anchor that pulls the
    student's final clean trajectory back toward a frozen base policy
    (``anchor_planner``, goal-free IL base) denoising the SAME student state:
        L += anchor_weight * mean|| x0_student_final - x0_anchor_final ||^2
    This regularises extrapolated / off-manifold rollouts toward the base
    manifold without adding a teacher.  ``anchor_planner`` is injected by the
    agent; if it is None the anchor term is a no-op (logged as 0).

variant == 'phf'  (方案三: PHF 特权隐状态流)
    Keep the on-policy OPD loss and ADD a privileged hidden-flow term that makes
    the student's (goal-FREE) DiT trunk mimic the routed teacher's (goal-
    MODULATED) DiT trunk, both the per-step hidden state and its step-to-step
    change (the "flow"):
        L += phf_weight      * mean|| h_student - h_teacher ||^2
           + phf_flow_weight * mean|| dh_student - dh_teacher ||^2
    Hidden states are captured non-invasively with forward hooks on the last
    transformer block of ``.model`` (no planner/DiT source change).  Student and
    teacher share the DiT config (``small``, D=384), so the alignment needs no
    projection layer.

All diagnostics from the baseline trainer are preserved, and the three extra
scalars ``anchor_loss`` / ``phf_hidden_loss`` / ``phf_flow_loss`` are ALWAYS
emitted (0 when the variant does not use them) so the DDP-logged key set stays
rank-symmetric.
"""

from typing import Dict, List, Optional

import math

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_dit_scene_router_goal_distill_trainer import (
    ReCogDriveDiTSceneRouterGoalDistillTrainer,
)

_LOG_2PI_E = math.log(2.0 * math.pi * math.e)

RESEARCH_VARIANTS = ("kl", "anchor", "phf")


class ReCogDriveGoalDistillResearchTrainer(ReCogDriveDiTSceneRouterGoalDistillTrainer):
    """Goal-teacher OPD with one of three research variants (kl / anchor / phf)."""

    def __init__(
        self,
        *args,
        variant: str = "kl",
        anchor_weight: float = 0.3,
        phf_weight: float = 0.1,
        phf_flow_weight: float = 0.1,
        collect_viz: bool = True,
        **kwargs,
    ):
        super().__init__(*args, collect_viz=collect_viz, **kwargs)
        if variant not in RESEARCH_VARIANTS:
            raise ValueError(f"variant must be one of {RESEARCH_VARIANTS}, got {variant!r}")
        self.variant = variant
        self.anchor_weight = float(anchor_weight)
        self.phf_weight = float(phf_weight)
        self.phf_flow_weight = float(phf_flow_weight)

        # 方案一: logprob-KL == match the transition mean. Force it here so the
        # variant is self-contained regardless of what match_target the run
        # script passes (a robust safety net over the config default).
        if self.variant == "kl":
            self.match_target = "mu"

        # Injected by the research agent for variant=='anchor' (frozen IL base).
        # Stored as a plain attribute; the agent keeps it out of the DDP tree.
        self.anchor_planner: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------- phf helpers
    @staticmethod
    def _last_block(planner):
        """The last DiT transformer block whose output is the (B, S, D) hidden."""
        return planner.model.transformer_blocks[-1]

    @staticmethod
    def _capture_hook(store: Dict[str, torch.Tensor], key: str):
        def _hook(_module, _inp, out):
            store[key] = out[0] if isinstance(out, (tuple, list)) else out
        return _hook

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

        gt_traj = action_input.action.float()          # (B, H, 3), raw metres
        goal = gt_traj[:, -1, :].contiguous()          # (B, 3)

        use_exopd = ref_planner is not None and abs(self.exopd_lambda - 1.0) > 1e-6
        use_x0 = self.match_target == "x0"
        use_anchor = self.variant == "anchor" and self.anchor_planner is not None
        use_phf = self.variant == "phf"

        # 1) on-policy detached student DDIM chain (goal-free, by construction)
        with torch.no_grad():
            vl_s0, his_s0, ego_s0 = self._encode(
                student_planner, vl_features, his_traj, ego_status, student_dtype
            )
            chain = self._sample_chain(
                student_planner, vl_s0, his_s0, ego_s0, batch_size, device, student_dtype
            )
        num_steps = chain.shape[1] - 1
        ddim_t_list = [int(student_planner.ddim_t[i].item()) for i in range(num_steps)]

        # 2) teacher / ref / anchor encodings (frozen, float32)
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
            anchor_encoding = (
                self._encode(self.anchor_planner, vl_features, his_traj, ego_status, torch.float32)
                if use_anchor else None
            )

        # 3) student encodings (with grad)
        vl_s, his_s, ego_s = self._encode(
            student_planner, vl_features, his_traj, ego_status, student_dtype
        )

        resolved = [self._resolve_bucket(b) for b in bucket_per_sample]
        bucket_to_indices: Dict[str, List[int]] = {}
        for i, b in enumerate(resolved):
            bucket_to_indices.setdefault(b, []).append(i)

        total_loss = vl_features.new_zeros(())
        per_bucket_step_losses: Dict[str, list] = {b: [] for b in self.bucket_names}
        sigma_list = []
        last_student_x0 = None

        # metric accumulators
        entropy_steps: List[torch.Tensor] = []
        x0_gap_steps: List[torch.Tensor] = []
        gauss_kl_steps: List[torch.Tensor] = []
        x0_gap_final = vl_features.new_zeros(())
        goal_effect_per_bucket: Dict[str, torch.Tensor] = {}
        last_teacher_x0_full = torch.zeros(
            batch_size, chain.shape[2], chain.shape[3], device=device
        )

        # variant accumulators
        anchor_loss = vl_features.new_zeros(())
        phf_hidden_loss = vl_features.new_zeros(())
        phf_flow_loss = vl_features.new_zeros(())

        # ---- PHF: non-invasive hidden-state capture via forward hooks ----
        hook_store: Dict[str, torch.Tensor] = {}
        hook_handles = []
        prev_h_s_full: Optional[torch.Tensor] = None
        prev_h_t_full: Optional[torch.Tensor] = None
        if use_phf:
            hook_handles.append(
                self._last_block(student_planner).register_forward_hook(
                    self._capture_hook(hook_store, "student")
                )
            )
            for name, teacher in teacher_planners.items():
                hook_handles.append(
                    self._last_block(teacher).register_forward_hook(
                        self._capture_hook(hook_store, f"teacher::{name}")
                    )
                )

        try:
            for step in range(num_steps):
                is_last = step == num_steps - 1
                z_t = chain[:, step].to(student_dtype)
                t_batch = student_planner.make_timesteps(batch_size, ddim_t_list[step], device)
                idx_batch = student_planner.make_timesteps(batch_size, step, device)

                mu_s, logvar_s, x0_s = student_planner.p_mean_variance(
                    z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False
                )
                # student hidden for this step (grad-connected), full batch
                h_s_full = hook_store.get("student") if use_phf else None

                sigma_s = self._safe_sigma(logvar_s, self.min_sigma, student_dtype).detach()
                sigma2 = sigma_s.float().pow(2).clamp(min=1e-6)
                sigma_list.append(sigma_s.detach().float().mean())
                student_side = (x0_s if use_x0 else mu_s).float()

                traj_dims = float(x0_s.shape[1] * x0_s.shape[2])
                entropy_steps.append(
                    ((0.5 * (_LOG_2PI_E + sigma2.log())).mean(dim=(1, 2)) * traj_dims)
                    .mean()
                    .detach()
                )

                ref_side_full = None
                if use_exopd:
                    with torch.no_grad():
                        mu_r, _, x0_r = ref_planner.p_mean_variance(
                            z_t.float(), t_batch, idx_batch,
                            ref_encoding[0], ref_encoding[1], ref_encoding[2],
                            deterministic=True,
                        )
                        ref_side_full = (x0_r if use_x0 else mu_r).detach()

                # full-batch teacher hidden buffer for this step (PHF)
                h_t_full = (
                    torch.zeros_like(h_s_full) if (use_phf and h_s_full is not None) else None
                )

                step_loss = vl_features.new_zeros(())
                n_used = 0
                step_gap_sum = vl_features.new_zeros(())
                step_kl_sum = vl_features.new_zeros(())
                for bucket, indices in bucket_to_indices.items():
                    enc = teacher_encodings[bucket]
                    teacher = teacher_planners[bucket]
                    sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                    z_sel = z_t[sel].float()
                    t_sel = t_batch[sel]
                    idx_sel = idx_batch[sel]
                    goal_sel = goal[sel]
                    with torch.no_grad():
                        with teacher.goal_context(goal_sel):
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

                        # PHF: grab the goal-bound teacher hidden right now, BEFORE
                        # the goal-off liveness call below overwrites the hook.
                        if h_t_full is not None:
                            h_t_full[sel] = hook_store[f"teacher::{bucket}"].to(h_t_full.dtype)

                        gap_m = self._waypoint_gap_m(student_planner, x0_s[sel], x0_t)
                        step_gap_sum = step_gap_sum + gap_m * len(indices)
                        kl = (
                            (mu_s[sel].float() - mu_t.float()).pow(2)
                            .div(2.0 * sigma2[sel])
                            .sum(dim=(1, 2))
                            .mean()
                        )
                        step_kl_sum = step_kl_sum + kl * len(indices)

                        if is_last:
                            last_teacher_x0_full[sel] = x0_t.float()
                            _, _, x0_t_off = teacher.p_mean_variance(
                                z_sel, t_sel, idx_sel,
                                enc[0][sel], enc[1][sel], enc[2][sel],
                                deterministic=True,
                            )
                            goal_effect_per_bucket[bucket] = self._waypoint_gap_m(
                                student_planner, x0_t, x0_t_off
                            ).detach()

                    diff = student_side[sel] - target
                    loss_per_sample = diff.pow(2).div(2.0 * sigma2[sel]).sum(dim=(1, 2))
                    per_bucket_step_losses[bucket].append(loss_per_sample.mean().detach())
                    step_loss = step_loss + loss_per_sample.sum()
                    n_used += len(indices)

                total_loss = total_loss + step_loss / max(n_used, 1)
                x0_gap_steps.append((step_gap_sum / max(n_used, 1)).detach())
                gauss_kl_steps.append((step_kl_sum / max(n_used, 1)).detach())

                # ---- PHF hidden + flow (student keeps grad; teacher detached) ----
                if use_phf and h_s_full is not None and h_t_full is not None:
                    h_s = h_s_full.float()
                    h_t = h_t_full.float().detach()
                    phf_hidden_loss = phf_hidden_loss + (h_s - h_t).pow(2).mean()
                    if prev_h_s_full is not None:
                        dh_s = h_s - prev_h_s_full
                        dh_t = h_t - prev_h_t_full
                        phf_flow_loss = phf_flow_loss + (dh_s - dh_t).pow(2).mean()
                    prev_h_s_full = h_s
                    prev_h_t_full = h_t

                if is_last:
                    last_student_x0 = x0_s
                    x0_gap_final = x0_gap_steps[-1]

                    # ---- Manifold Anchor: pull final x0 toward the base policy ----
                    if use_anchor:
                        with torch.no_grad():
                            _, _, x0_anchor = self.anchor_planner.p_mean_variance(
                                z_t.float(), t_batch, idx_batch,
                                anchor_encoding[0], anchor_encoding[1], anchor_encoding[2],
                                deterministic=True,
                            )
                        anchor_loss = (x0_s.float() - x0_anchor.float().detach()).pow(2).mean()
        finally:
            for h in hook_handles:
                h.remove()

        distill_loss = total_loss / num_steps
        pred_traj_s = student_planner.denorm_odo(last_student_x0.float())
        smooth_loss = self._jerk_loss(pred_traj_s)

        phf_hidden_loss = phf_hidden_loss / max(num_steps, 1)
        phf_flow_loss = phf_flow_loss / max(num_steps - 1, 1)

        loss = distill_loss + self.smooth_weight * smooth_loss
        if use_anchor:
            loss = loss + self.anchor_weight * anchor_loss
        if use_phf:
            loss = loss + self.phf_weight * phf_hidden_loss + self.phf_flow_weight * phf_flow_loss

        # See baseline: keep eta in the DDP graph with a finite zero coefficient
        # (eta_logit is atanh(1.0)=+inf under EtaFixed; plain *0 -> inf*0 = NaN).
        if hasattr(student_planner, "eta") and hasattr(student_planner.eta, "eta_logit"):
            eta_logit = student_planner.eta.eta_logit
            loss = loss + torch.nan_to_num(eta_logit, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        if not torch.isfinite(loss):
            loss = last_student_x0.float().sum() * 0.0

        with torch.no_grad():
            err_xy = (pred_traj_s[..., :2] - gt_traj[..., :2]).norm(dim=-1)  # (B, H)
            student_ade = err_xy.mean()
            student_fde = err_xy[:, -1].mean()
            fde_per_bucket = {
                bucket: err_xy[torch.as_tensor(idxs, device=device), -1].mean().detach()
                for bucket, idxs in bucket_to_indices.items()
            }

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
            "x0_gap_m": torch.stack(x0_gap_steps).mean(),
            "x0_gap_final_m": x0_gap_final,
            "gauss_kl_mean": torch.stack(gauss_kl_steps).mean(),
            "entropy_mean": torch.stack(entropy_steps).mean(),
            "student_fde_gt_m": student_fde.detach(),
            "student_ade_gt_m": student_ade.detach(),
            "teacher_goal_effect_m": (
                torch.stack(list(goal_effect_per_bucket.values())).mean()
                if goal_effect_per_bucket
                else torch.tensor(0.0, device=device)
            ),
            # research-variant scalars (always present -> rank-symmetric)
            "anchor_loss": anchor_loss.detach(),
            "phf_hidden_loss": phf_hidden_loss.detach(),
            "phf_flow_loss": phf_flow_loss.detach(),
        }
        for i, ent in enumerate(entropy_steps):
            data[f"entropy_step_{i}"] = ent

        zero = torch.tensor(0.0, device=device)
        for bucket in self.bucket_names:
            values = per_bucket_step_losses[bucket]
            data[f"kl_{bucket}_mean"] = torch.stack(values).mean().detach() if values else zero
            data[f"n_samples_{bucket}"] = torch.tensor(
                float(len(bucket_to_indices.get(bucket, []))), device=device
            )
            data[f"fde_gt_{bucket}"] = fde_per_bucket.get(bucket, zero)
            data[f"goal_effect_{bucket}"] = goal_effect_per_bucket.get(bucket, zero)

        if self.collect_viz:
            with torch.no_grad():
                self.last_viz = {
                    "gt_traj": gt_traj.detach().cpu(),
                    "goal": goal.detach().cpu(),
                    "student_traj": pred_traj_s.detach().float().cpu(),
                    "teacher_traj": student_planner.denorm_odo(last_teacher_x0_full)
                    .detach().float().cpu(),
                    "buckets": list(resolved),
                }

        return BatchFeature(data=data)

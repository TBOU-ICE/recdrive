"""Distributional GoalBridge OPD for privileged driving teachers.

This extends the v1 GoalBridge loss with trajectory-support distillation.
For each scene we build a shared candidate set containing stochastic trajectories
sampled from both the privileged teacher and the current student. Teacher and
student then score the *same diffusion chains* with their own conditional policy
likelihoods. A categorical JSD over the candidate set transfers the teacher's
local trajectory support instead of only matching one reverse-process mean.

The original dense per-DDIM-step reverse-KL remains intact and is still useful
for precise within-mode alignment. The new support loss adds cross-mode coverage.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from .recogdrive_goalbridge_distill_trainer import ReCogDriveGoalBridgeDistillTrainer


class ReCogDriveDistributionalGoalBridgeDistillTrainer(ReCogDriveGoalBridgeDistillTrainer):
    def __init__(
        self,
        *args,
        support_weight: float = 0.25,
        support_teacher_candidates: int = 4,
        support_student_candidates: int = 4,
        support_temperature: float = 0.5,
        support_logprob_clip_min: float = -5.0,
        support_logprob_clip_max: float = 2.0,
        support_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.support_weight = float(support_weight)
        self.support_teacher_candidates = int(support_teacher_candidates)
        self.support_student_candidates = int(support_student_candidates)
        self.support_temperature = float(support_temperature)
        self.support_logprob_clip_min = float(support_logprob_clip_min)
        self.support_logprob_clip_max = float(support_logprob_clip_max)
        self.support_eps = float(support_eps)
        if self.support_teacher_candidates < 1:
            raise ValueError("support_teacher_candidates must be >= 1")
        if self.support_student_candidates < 0:
            raise ValueError("support_student_candidates must be >= 0")
        if self.support_temperature <= 0:
            raise ValueError("support_temperature must be > 0")

    @staticmethod
    def _repeat_batch(x: torch.Tensor, repeats: int) -> torch.Tensor:
        return x.repeat_interleave(repeats, dim=0)

    def _chain_scores(
        self,
        planner,
        vl: torch.Tensor,
        his: torch.Tensor,
        ego: torch.Tensor,
        chains: torch.Tensor,
        batch_size: int,
        num_candidates: int,
        goal: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return normalized chain log-likelihoods with shape [B, K]."""
        planner_dtype = next(planner.parameters()).dtype
        vl = vl.to(planner_dtype)
        his = his.to(planner_dtype)
        ego = ego.to(planner_dtype)
        chains = chains.to(planner_dtype)
        if goal is not None:
            goal = goal.to(planner_dtype)
            ctx = planner.goal_context(goal)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()
        with ctx:
            logp = planner.get_logprobs(vl, his, ego, chains, deterministic=False)
        # get_logprobs => [B*K*n_steps, H, D]. Normalize by dimensionality so the
        # temperature is stable when denoising step count / horizon changes.
        n_steps = chains.shape[1] - 1
        logp = logp.clamp(self.support_logprob_clip_min, self.support_logprob_clip_max)
        logp = logp.view(batch_size, num_candidates, n_steps, chains.shape[2], chains.shape[3])
        return logp.mean(dim=(2, 3, 4))

    def _jsd_from_logits(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.support_temperature
        p_t = F.softmax(teacher_logits / t, dim=-1)
        p_s = F.softmax(student_logits / t, dim=-1)
        m = 0.5 * (p_t + p_s)
        eps = self.support_eps
        jsd = 0.5 * (
            (p_t * ((p_t + eps).log() - (m + eps).log())).sum(dim=-1)
            + (p_s * ((p_s + eps).log() - (m + eps).log())).sum(dim=-1)
        )
        entropy_t = -(p_t * (p_t + eps).log()).sum(dim=-1)
        entropy_s = -(p_s * (p_s + eps).log()).sum(dim=-1)
        return jsd.mean(), entropy_t.mean(), entropy_s.mean()

    def _support_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        vl_features: torch.Tensor,
        action_input,
        bucket_per_sample: List[str],
    ) -> BatchFeature:
        if self.support_weight <= 0:
            z = vl_features.new_zeros((), dtype=torch.float32)
            return BatchFeature(data={
                "support_jsd_loss": z,
                "support_teacher_entropy": z,
                "support_student_entropy": z,
                "support_num_candidates": z,
            })

        device = vl_features.device
        his = action_input.his_traj
        ego = action_input.status_feature
        gt_traj = action_input.action.float()
        gt_goal = gt_traj[:, -1, :].contiguous()
        resolved = [self._resolve_bucket(b) for b in bucket_per_sample]

        bucket_to_indices: Dict[str, List[int]] = {}
        for i, bucket in enumerate(resolved):
            bucket_to_indices.setdefault(bucket, []).append(i)

        total_jsd = vl_features.new_zeros((), dtype=torch.float32)
        total_te = vl_features.new_zeros((), dtype=torch.float32)
        total_se = vl_features.new_zeros((), dtype=torch.float32)
        total_n = 0
        total_k = 0

        for bucket, indices in bucket_to_indices.items():
            sel = torch.as_tensor(indices, device=device, dtype=torch.long)
            b = len(indices)
            teacher = teacher_planners[bucket]
            teacher.eval()

            vl_b = vl_features[sel]
            his_b = his[sel]
            ego_b = ego[sel]
            goal_b = gt_goal[sel]

            # Student's deployable conditioning. Keep gradients for likelihood
            # scoring, while candidate sampling itself stays detached.
            pred_goal_b, _ = student_planner.predict_goal(vl_b, his_b, ego_b)

            kt = self.support_teacher_candidates
            ks = self.support_student_candidates
            candidate_groups = []

            # Privileged teacher modes.
            with torch.no_grad():
                vl_t = self._repeat_batch(vl_b.float(), kt)
                his_t = self._repeat_batch(his_b.float(), kt)
                ego_t = self._repeat_batch(ego_b.float(), kt)
                goal_t = self._repeat_batch(goal_b.float(), kt)
                with teacher.goal_context(goal_t):
                    teacher_chains, _ = teacher.sample_chain(vl_t, his_t, ego_t, deterministic=False)
                teacher_chains = teacher_chains.view(b, kt, *teacher_chains.shape[1:])
                candidate_groups.append(teacher_chains)

            # Current student modes. Including them in the shared support set lets
            # the privileged teacher explicitly down-weight student-only bad modes.
            if ks > 0:
                with torch.no_grad():
                    vl_samp = self._repeat_batch(vl_b, ks)
                    his_samp = self._repeat_batch(his_b, ks)
                    ego_samp = self._repeat_batch(ego_b, ks)
                    goal_samp = self._repeat_batch(pred_goal_b.detach(), ks)
                    with student_planner.goal_context(goal_samp):
                        student_chains, _ = student_planner.sample_chain(
                            vl_samp, his_samp, ego_samp, deterministic=False
                        )
                    student_chains = student_chains.view(b, ks, *student_chains.shape[1:])
                    candidate_groups.append(student_chains)

            candidates = torch.cat(candidate_groups, dim=1).detach()
            k = candidates.shape[1]
            chains_flat = candidates.flatten(0, 1)
            vl_rep = self._repeat_batch(vl_b, k)
            his_rep = self._repeat_batch(his_b, k)
            ego_rep = self._repeat_batch(ego_b, k)
            gt_goal_rep = self._repeat_batch(goal_b, k)
            pred_goal_rep = self._repeat_batch(pred_goal_b, k)

            # Teacher target distribution is frozen; student scores carry grads.
            with torch.no_grad():
                teacher_logits = self._chain_scores(
                    teacher, vl_rep.float(), his_rep.float(), ego_rep.float(), chains_flat.float(),
                    batch_size=b, num_candidates=k, goal=gt_goal_rep.float(),
                )
            student_logits = self._chain_scores(
                student_planner, vl_rep, his_rep, ego_rep, chains_flat,
                batch_size=b, num_candidates=k, goal=pred_goal_rep,
            )

            jsd, ent_t, ent_s = self._jsd_from_logits(teacher_logits.detach(), student_logits)
            total_jsd = total_jsd + jsd * b
            total_te = total_te + ent_t.detach() * b
            total_se = total_se + ent_s.detach() * b
            total_n += b
            total_k += k * b

        denom = max(total_n, 1)
        return BatchFeature(data={
            "support_jsd_loss": total_jsd / denom,
            "support_teacher_entropy": total_te / denom,
            "support_student_entropy": total_se / denom,
            "support_num_candidates": torch.tensor(float(total_k / denom), device=device),
        })

    def compute_loss(
        self,
        student_planner,
        teacher_planners: Dict[str, torch.nn.Module],
        ref_planner: Optional[torch.nn.Module],
        vl_features: torch.Tensor,
        action_input,
        bucket_per_sample: List[str],
        anchor_planner: Optional[torch.nn.Module] = None,
    ) -> BatchFeature:
        local = super().compute_loss(
            student_planner=student_planner,
            teacher_planners=teacher_planners,
            ref_planner=ref_planner,
            vl_features=vl_features,
            action_input=action_input,
            bucket_per_sample=bucket_per_sample,
            anchor_planner=anchor_planner,
        )
        support = self._support_loss(
            student_planner=student_planner,
            teacher_planners=teacher_planners,
            vl_features=vl_features,
            action_input=action_input,
            bucket_per_sample=bucket_per_sample,
        )
        support_loss = support.support_jsd_loss
        local.loss = local.loss + self.support_weight * support_loss
        local["support_jsd_loss"] = support_loss.detach()
        local["weighted_support_jsd_loss"] = (self.support_weight * support_loss).detach()
        local["support_teacher_entropy"] = support.support_teacher_entropy.detach()
        local["support_student_entropy"] = support.support_student_entropy.detach()
        local["support_num_candidates"] = support.support_num_candidates.detach()
        return local

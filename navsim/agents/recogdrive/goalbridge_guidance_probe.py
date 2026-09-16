"""Offline probe: is the routed teacher's guidance sane at the student's states?

Pure inference (``torch.no_grad`` throughout).  Nothing in this module is
imported by the training path, so it cannot perturb an OPD/SFT run.

At every DDIM step the probe produces up to four x0 predictions, all
denormalised to metres so they can be drawn on one BEV axis:

==================  ==========================  =========================  ======
key                 z_t source                  goal conditioning          colour
==================  ==========================  =========================  ======
``student_pred``    student DDIM chain          student goal               blue
``teacher_gt``      student DDIM chain          GT goal                    red
``teacher_pred``    student DDIM chain          student predicted goal     dark yellow
``teacher_qgt``     forward-noised GT           GT goal                    green
==================  ==========================  =========================  ======

``teacher_gt`` is the condition already drawn during training; the other three
are the controls that make it interpretable, each differing from it in exactly
one variable:

* ``teacher_gt`` vs ``teacher_pred`` -- same z_t, goal swapped.  Isolates the
  goal-conditioning conflict: the chain was rolled out under the predicted goal
  while the teacher is asked to denoise towards the GT goal.
* ``teacher_gt`` vs ``teacher_qgt`` -- same goal, z_t swapped for the teacher's
  own training marginal q(z_t | x0_gt).  Isolates "the student's z_t is
  off-distribution" from "the teacher is simply weak at this noise level".
* ``student_pred`` vs ``teacher_gt`` -- the gap the KD term actually pulls on.

Two per-step scalars are recorded alongside, because the BEV panels alone are
easy to misread:

* ``clamp_frac`` -- per-sample fraction of x0 elements sitting on the +-``denoised_clip_value``
  boundary.  x0 is reconstructed as ``(z - sqrt(1-a)*eps_hat)/sqrt(a)``, so at
  early DDIM steps (small ``a``) any eps_hat error is amplified and then
  saturated by the clamp.  A high value means the ragged trajectory is a
  truncation artefact, not bad guidance.
* ``z_second_moment_ratio`` -- ``E[z_t^2]`` divided by the analytic q marginal
  ``a * E[x0_norm^2] + (1 - a)``.  A value away from 1 is direct evidence that
  the student's chain leaves the noise scale the teacher was trained on, with no
  eyeballing involved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from navsim.agents.recogdrive.recogdrive_dit_scene_router_distill_trainer import (
    ReCogDriveDiTSceneRouterDistillTrainer,
)

# Conditions that exist only when the student has a usable goal head.
_PRED_GOAL_CONDITIONS = ("teacher_pred",)
ALL_CONDITIONS = ("student_pred", "teacher_gt", "teacher_pred", "teacher_qgt")


class GoalBridgeGuidanceProbe:
    """Evaluate teacher/student x0 under matched and mismatched conditions.

    Parameters
    ----------
    agent:
        A built and ``initialize()``-d GoalBridge agent.  Its ``action_head``
        (student), ``teacher_planners`` and ``scene_router_trainer`` are reused
        as-is, so the probed states match training exactly -- including the
        ``min_sigma`` floor applied while sampling the chain.
    student_goal_source:
        ``"pred"`` conditions the student chain on its own predicted goal (the
        deployable GoalBridge student).  ``"none"`` rolls the chain out
        goal-free, which is the right setting for a base IL checkpoint whose
        goal head and goal encoder were never trained: the goal encoder's last
        layer is zero-initialised, so for ``goal_mode="adaln"`` an untrained
        goal embedding is added as zero anyway and "goal-free" is what the
        model already does.  ``teacher_pred`` is skipped in this mode.
    seed:
        Seeds both the student chain noise and the forward-noising epsilon, so
        two checkpoints are compared over the same noise draw.
    """

    def __init__(
        self,
        agent,
        student_goal_source: str = "pred",
        seed: int = 0,
    ):
        if student_goal_source not in ("pred", "none"):
            raise ValueError(
                f"student_goal_source must be 'pred' or 'none', got {student_goal_source!r}"
            )
        self.agent = agent
        self.student_goal_source = student_goal_source
        self.seed = int(seed)
        self.student = agent.action_head
        self.teachers = agent.teacher_planners
        self.trainer = agent.scene_router_trainer
        self._batch_index = 0
        self.student.eval()
        for teacher in self.teachers.values():
            teacher.eval()

    # ------------------------------------------------------------------ utils
    @property
    def _encode(self):
        return ReCogDriveDiTSceneRouterDistillTrainer._encode

    @staticmethod
    def _clamp_fraction(x0: torch.Tensor, clip_value: float) -> torch.Tensor:
        """Per-sample fraction of x0 elements pinned to the +-clip boundary.

        Reported per sample rather than per batch so a single ragged panel can be
        attributed to truncation without averaging that evidence away.
        """
        if clip_value <= 0:
            return torch.full((x0.shape[0],), float("nan"))
        on_boundary = x0.float().abs() >= (clip_value - 1e-4)
        return on_boundary.float().mean(dim=(1, 2)).cpu()

    def _unpack(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Mirror ``ReCogDriveGoalBridgeAgent.forward`` feature handling."""
        model_dtype = next(self.student.parameters()).dtype
        last_hidden_state = features["last_hidden_state"].cuda().to(model_dtype)
        status_feature = features["status_feature"].cuda()
        history_trajectory = features["history_trajectory"].cuda()
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        history_flat = history_trajectory.view(history_trajectory.size(0), -1)
        gt_traj = targets["trajectory"].cuda().float()
        return (
            last_hidden_state,
            history_flat.to(model_dtype),
            status_feature.to(model_dtype),
            gt_traj,
        )

    # -------------------------------------------------------------------- run
    @torch.no_grad()
    def run_batch(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        tokens_list: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Probe one collated batch.  Returns CPU tensors in metres."""
        student = self.student
        vl_features, his_traj, ego_status, gt_traj = self._unpack(features, targets)
        batch_size = vl_features.shape[0]
        device = vl_features.device
        student_dtype = next(student.parameters()).dtype
        gt_goal = gt_traj[:, -1, :].contiguous()
        clip_value = float(getattr(student, "denoised_clip_value", 1.0))

        vl_s, his_s, ego_s = self._encode(
            student, vl_features, his_traj, ego_status, student_dtype
        )

        # Student goal. A base IL checkpoint has an untrained goal head, so the
        # caller asks for "none" and the chain is rolled out goal-free.
        if self.student_goal_source == "pred":
            pred_goal, _ = student.predict_goal_from_encoded(vl_s, his_s, ego_s)
            pred_goal = pred_goal.detach()
            student_goal: Optional[torch.Tensor] = pred_goal
        else:
            pred_goal = None
            student_goal = None

        # Same sampler as training (including the min_sigma floor). The seed is
        # derived from the batch index so successive batches draw independent
        # noise while any two checkpoints still see an identical sequence.
        batch_seed = self.seed + 1000 * self._batch_index
        torch.manual_seed(batch_seed)
        with student.goal_context(student_goal):
            chain = self.trainer._sample_chain(
                student, vl_s, his_s, ego_s, batch_size, device, student_dtype
            )
        num_steps = chain.shape[1] - 1
        ddim_t_list = [int(student.ddim_t[i].item()) for i in range(num_steps)]

        teacher_encodings = {
            name: self._encode(teacher, vl_features, his_traj, ego_status, torch.float32)
            for name, teacher in self.teachers.items()
        }
        resolved = [self.trainer._resolve_bucket(b) for b in self._buckets(tokens_list, batch_size)]
        bucket_to_indices: Dict[str, List[int]] = {}
        for i, bucket in enumerate(resolved):
            bucket_to_indices.setdefault(bucket, []).append(i)

        # Forward-noising epsilon is drawn once and reused across steps so the
        # green curve varies only through the noise *level*, not a fresh draw.
        gen = torch.Generator(device=device).manual_seed(batch_seed + 1)
        gt_norm = student.norm_odo(gt_traj).float()
        eps_q = torch.randn(gt_norm.shape, device=device, generator=gen)
        gt_second_moment = gt_norm.pow(2).mean()

        conditions = self._active_conditions()
        steps: List[Dict[str, Any]] = []

        for step in range(num_steps):
            z_t = chain[:, step].to(student_dtype)
            t_batch = student.make_timesteps(batch_size, ddim_t_list[step], device)
            idx_batch = student.make_timesteps(batch_size, step, device)
            alpha_t = student.extract(student.ddim_alphas, idx_batch, z_t.shape).float()

            # Teacher training marginal q(z_t | x0_gt) at the very same level.
            z_q = alpha_t.sqrt() * gt_norm + (1.0 - alpha_t).clamp(min=0.0).sqrt() * eps_q

            x0: Dict[str, torch.Tensor] = {}

            with student.goal_context(student_goal):
                _, _, x0_student = student.p_mean_variance(
                    z_t, t_batch, idx_batch, vl_s, his_s, ego_s, deterministic=False
                )
            x0["student_pred"] = x0_student.float()

            for key in conditions:
                if key == "student_pred":
                    continue
                x0[key] = torch.zeros_like(x0_student, dtype=torch.float32)

            for bucket, indices in bucket_to_indices.items():
                sel = torch.as_tensor(indices, device=device, dtype=torch.long)
                teacher = self.teachers[bucket]
                enc = teacher_encodings[bucket]

                def _teacher_x0(z: torch.Tensor, goal: Optional[torch.Tensor]) -> torch.Tensor:
                    with teacher.goal_context(goal):
                        _, _, out = teacher.p_mean_variance(
                            z[sel].float(),
                            t_batch[sel],
                            idx_batch[sel],
                            enc[0][sel],
                            enc[1][sel],
                            enc[2][sel],
                            deterministic=True,
                        )
                    return out.float()

                x0["teacher_gt"][sel] = _teacher_x0(z_t, gt_goal[sel])
                x0["teacher_qgt"][sel] = _teacher_x0(z_q, gt_goal[sel])
                if "teacher_pred" in conditions:
                    x0["teacher_pred"][sel] = _teacher_x0(z_t, pred_goal[sel])

            record: Dict[str, Any] = {
                "step": step,
                "timestep": ddim_t_list[step],
                "alpha_bar": float(alpha_t.reshape(-1)[0]),
                "traj": {},
                "clamp_frac": {},
                "ade_m": {},
                "fde_m": {},
            }
            for key, value in x0.items():
                traj = student.denorm_odo(value).float()
                err = (traj[..., :2] - gt_traj[..., :2]).norm(dim=-1)
                record["traj"][key] = traj.cpu()
                record["clamp_frac"][key] = self._clamp_fraction(value, clip_value)
                record["ade_m"][key] = err.mean(dim=-1).cpu()
                record["fde_m"][key] = err[:, -1].cpu()

            # Off-distribution evidence that does not depend on reading a plot.
            expected = alpha_t.reshape(-1)[0] * gt_second_moment + (1.0 - alpha_t.reshape(-1)[0])
            record["z_second_moment_ratio"] = float(
                z_t.float().pow(2).mean() / expected.clamp(min=1e-8)
            )
            record["z_abs_max"] = float(z_t.float().abs().max())
            steps.append(record)

        self._batch_index += 1
        goal_err = (
            (pred_goal.float()[..., :2] - gt_goal[..., :2]).norm(dim=-1).cpu()
            if pred_goal is not None
            else torch.full((batch_size,), float("nan"))
        )

        return {
            "gt_traj": gt_traj.cpu(),
            "gt_goal": gt_goal.cpu(),
            "pred_goal": pred_goal.float().cpu() if pred_goal is not None else None,
            "goal_err_m": goal_err,
            "buckets": list(resolved),
            "tokens": list(tokens_list) if tokens_list is not None else None,
            "steps": steps,
            "conditions": list(conditions),
            "ddim_t": ddim_t_list,
            "clip_value": clip_value,
        }

    # --------------------------------------------------------------- helpers
    def _active_conditions(self) -> List[str]:
        if self.student_goal_source == "pred":
            return list(ALL_CONDITIONS)
        return [c for c in ALL_CONDITIONS if c not in _PRED_GOAL_CONDITIONS]

    def _buckets(self, tokens_list, batch_size: int) -> List[str]:
        return self.agent._buckets_for_tokens(tokens_list, batch_size)

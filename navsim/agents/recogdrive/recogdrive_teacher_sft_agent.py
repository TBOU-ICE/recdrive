"""Phase B: student SFT on offline teacher rollouts.

The student is the *same* ``PredictedGoalDiffusionPlanner`` the GoalBridge OPD
run trains, so the checkpoint this produces drops straight into ``STUDENT_CKPT``.

Two objectives are optimised, and they touch disjoint parameter sets:

``L_diff``
    The ordinary imitation loss (epsilon-MSE), but the target trajectory is a
    teacher rollout instead of the GT trajectory, and the goal the DiT is
    conditioned on is *that rollout's own endpoint*.  Conditioning on the
    rollout's endpoint rather than the GT endpoint makes the pair strictly
    self-consistent, which reproduces the teacher's own training task and drives
    the loss floor to ~0; conditioning on the GT endpoint would instead ask the
    student to predict the teacher's residual-vs-GT, which is largely
    unlearnable.  This is teacher forcing: the student's *own* predicted goal is
    deliberately not fed back here, because the goal head starts at zero output
    and would teach the DiT that the goal channel carries no information.

``L_goal``
    SmoothL1 from the goal head to the GT endpoint -- the same target the OPD
    trainer uses, so the head arrives at OPD already aligned.  It cannot
    contaminate the DiT: ``predict_goal_from_encoded`` detaches all three of its
    inputs, so this gradient reaches only ``goal_predictor``.

Goal corruption is enabled during SFT (see ``goal_dropout_p`` / ``goal_noise_p``).
Without it the DiT only ever sees a goal that is exactly the target's endpoint
and learns to treat it as a hard constraint, which breaks the moment OPD starts
feeding a predicted goal with real error.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.goal_cond import TRAINABLE_GOAL_MODES
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_predicted_goal_planner import (
    PredictedGoalDiffusionPlanner,
)
from navsim.agents.recogdrive.recogdrive_scene_router_agent import _normalize_token


class ReCogDriveTeacherSFTAgent(ReCogDriveAgent):
    """Predicted-goal student trained to imitate offline scenario-teacher rollouts."""

    def __init__(
        self,
        *args,
        teacher_rollout_dir: str = "",
        student_goal_mode: str = "adaln",
        goal_sincos_dim: int = 128,
        goal_hidden_dim: int = 1024,
        goal_use_heading: bool = False,
        goal_predictor_hidden_dim: int = 512,
        goal_predictor_dropout: float = 0.0,
        goal_loss_weight: float = 1.0,
        # Teacher recipe was 0.10/0.20/1.0m, calibrated for a model that always
        # receives a GT goal at eval.  The student will instead receive its own
        # predicted goal, whose error is larger, so widen the corruption.
        goal_dropout_p: float = 0.10,
        goal_noise_p: float = 0.30,
        goal_noise_std_xy: float = 2.0,
        goal_noise_std_heading: float = 0.0,
        goal_target: str = "gt",
        min_rollout_coverage: float = 0.90,
        goal_sensitivity_interval: int = 0,
        student_adaln_bound: float = 8.0,
        **kwargs,
    ):
        kwargs["dit_distill"] = False
        kwargs["opd"] = False
        kwargs["grpo"] = False
        super().__init__(*args, **kwargs)

        if goal_target not in ("gt", "teacher_canonical"):
            raise ValueError(f"goal_target must be 'gt' or 'teacher_canonical', got {goal_target!r}")
        if student_goal_mode not in TRAINABLE_GOAL_MODES:
            # With an untrainable mode ``_goal_embedding_for`` returns None and the
            # whole goal branch silently no-ops -- SFT would then train a goal-free
            # student and the OPD stage would start from a dead goal_encoder anyway.
            raise ValueError(
                f"student_goal_mode must be one of {TRAINABLE_GOAL_MODES}, got {student_goal_mode!r}. "
                "Teacher SFT exists to train the goal branch; a no-op mode defeats it."
            )
        self.goal_target = goal_target
        self.goal_loss_weight = float(goal_loss_weight)
        self.min_rollout_coverage = float(min_rollout_coverage)
        self.goal_sensitivity_interval = int(goal_sensitivity_interval)
        self._eval_batches = 0
        self._covered_total = 0
        self._seen_total = 0
        self._coverage_warmup_samples = 2048

        old = self.action_head
        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=8,
            grpo=False,
            input_embedding_dim=384 if self.dit_type == "small" else 1536,
            sampling_method=old.config.sampling_method,
        )
        cfg.vlm_size = self.vlm_size
        planner = PredictedGoalDiffusionPlanner(
            cfg,
            goal_mode=student_goal_mode,
            goal_sincos_dim=goal_sincos_dim,
            goal_hidden_dim=goal_hidden_dim,
            goal_use_heading=goal_use_heading,
            goal_predictor_hidden_dim=goal_predictor_hidden_dim,
            goal_predictor_dropout=goal_predictor_dropout,
            goal_dropout_p=goal_dropout_p,
            goal_noise_p=goal_noise_p,
            goal_noise_std_xy=goal_noise_std_xy,
            goal_noise_std_heading=goal_noise_std_heading,
        ).cuda()
        planner.load_state_dict(old.state_dict(), strict=False)
        self.action_head = planner
        for param in self.action_head.parameters():
            param.requires_grad = True

        student_dit = getattr(self.action_head, "model", None)
        if student_dit is not None and hasattr(student_dit, "set_adaln_bound"):
            student_dit.set_adaln_bound(float(student_adaln_bound))

        self._load_rollouts(teacher_rollout_dir)

        print(
            "[TeacherSFT] student=PredictedGoalDiffusionPlanner "
            f"mode={student_goal_mode} goal_target={goal_target} goal_w={goal_loss_weight:g} | "
            f"corruption clean={1 - goal_dropout_p - goal_noise_p:.2f} "
            f"noisy={goal_noise_p:.2f}@{goal_noise_std_xy:g}m masked={goal_dropout_p:.2f} | "
            f"rollouts={self.rollout_trajectories.shape[0]} K+1={self.rollout_trajectories.shape[1]}"
        )

    # ----------------------------------------------------------------- rollouts
    def _load_rollouts(self, rollout_dir: str) -> None:
        """Load every shard written by ``run_teacher_rollout_cache.py``.

        Kept on CPU in the main process: ``forward`` runs there, so dataloader
        workers never fork a copy.  ~100 MB for 200k scenes at K+1=5.
        """
        if not rollout_dir:
            raise ValueError("teacher_rollout_dir is required for teacher SFT.")
        shard_paths = sorted(glob.glob(os.path.join(rollout_dir, "teacher_rollout_shard_*.pt")))
        if not shard_paths:
            raise FileNotFoundError(f"No teacher_rollout_shard_*.pt under {rollout_dir!r}")

        tokens: List[str] = []
        trajectory_chunks: List[torch.Tensor] = []
        fde_chunks: List[torch.Tensor] = []
        num_samples: Optional[int] = None
        for path in shard_paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            shard_traj = shard["trajectories"]
            if shard_traj.shape[0] == 0:
                continue
            if num_samples is None:
                num_samples = shard_traj.shape[1]
            elif shard_traj.shape[1] != num_samples:
                raise ValueError(
                    f"shard {path} has {shard_traj.shape[1]} samples/scene, expected {num_samples}; "
                    "the shards were not generated by one run."
                )
            tokens.extend(shard["tokens"])
            trajectory_chunks.append(shard_traj.float())
            fde_chunks.append(shard["fde_to_gt"].float())

        if not trajectory_chunks:
            raise RuntimeError(f"All shards under {rollout_dir!r} are empty.")

        trajectories = torch.cat(trajectory_chunks, dim=0)
        fde = torch.cat(fde_chunks, dim=0)

        # Shards are generated by disjoint ranks, but dedupe anyway so a rerun
        # with a different world size cannot silently double-weight scenes.
        index: Dict[str, int] = {}
        for i, token in enumerate(tokens):
            index.setdefault(token, i)
        if len(index) != len(tokens):
            keep = torch.as_tensor(sorted(index.values()), dtype=torch.long)
            trajectories = trajectories.index_select(0, keep)
            fde = fde.index_select(0, keep)
            index = {tokens[int(j)]: i for i, j in enumerate(keep)}

        self.rollout_index = index
        self.rollout_trajectories = trajectories  # (N, K+1, 8, 3), index 0 = canonical
        self.rollout_fde_to_gt = fde
        print(
            f"[TeacherSFT] loaded {len(shard_paths)} shard(s) from {rollout_dir}: "
            f"{trajectories.shape[0]} scenes, canonical FDE-to-GT mean={fde.mean().item():.3f}m"
        )

    def _update_coverage(self, covered: int, batch_size: int) -> float:
        """Track the cumulative rollout hit rate and fail if the cache is wrong.

        Deliberately cumulative rather than per-batch: at 95% real coverage a
        16-sample batch misses two scenes often enough that a per-batch threshold
        would abort a perfectly healthy run.  The warmup lets the rate settle
        before it can trip.
        """
        self._covered_total += covered
        self._seen_total += batch_size
        rate = self._covered_total / max(self._seen_total, 1)
        if self._seen_total >= self._coverage_warmup_samples and rate < self.min_rollout_coverage:
            raise RuntimeError(
                f"Teacher rollout coverage {rate:.1%} over {self._seen_total} samples is below "
                f"min_rollout_coverage {self.min_rollout_coverage:.1%}; the rollout cache does "
                "not cover the dataset this run trains on."
            )
        return rate

    def _lookup_rollouts(
        self,
        tokens_list: Optional[Sequence[str]],
        batch_size: int,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rows, trajectories)`` for the batch entries that have a rollout.

        During training a random one of the ``K + 1`` rollouts is drawn per
        sample, so a scene contributes a different self-consistent
        ``(goal, trajectory)`` pair on each visit and the student sees several
        points of goal space per scene.  Validation always uses the canonical
        rollout so the reported loss is comparable across epochs.
        """
        if tokens_list is None:
            return torch.empty(0, dtype=torch.long), torch.empty(0)

        rows: List[int] = []
        picks: List[int] = []
        for i, token in enumerate(tokens_list[:batch_size]):
            norm = _normalize_token(token)
            position = self.rollout_index.get(norm) if norm else None
            if position is None:
                continue
            rows.append(i)
            picks.append(position)

        if not rows:
            return torch.empty(0, dtype=torch.long), torch.empty(0)

        picked = self.rollout_trajectories.index_select(0, torch.as_tensor(picks, dtype=torch.long))
        if training:
            choice = torch.randint(0, picked.shape[1], (picked.shape[0],))
            trajectories = picked[torch.arange(picked.shape[0]), choice]
        else:
            trajectories = picked[:, 0]
        return torch.as_tensor(rows, dtype=torch.long), trajectories

    # ------------------------------------------------------------------ forward
    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()
        if not self.cache_hidden_state:
            raise RuntimeError("Teacher SFT expects cache_hidden_state=True.")

        planner = self.action_head
        model_dtype = next(planner.parameters()).dtype

        history = features["history_trajectory"].cuda()
        status = features["status_feature"].cuda()
        last_hidden = features["last_hidden_state"].cuda()
        if history.ndim == 2:
            history = history.unsqueeze(0)
        if status.ndim == 1:
            status = status.unsqueeze(0)
        if last_hidden.ndim == 2:
            last_hidden = last_hidden.unsqueeze(0)
        history_flat = history.view(history.size(0), -1)
        last_hidden = last_hidden.to(model_dtype)
        batch_size = last_hidden.shape[0]

        if not self.training:
            return self._eval_forward(
                planner, last_hidden, history_flat, status, targets, model_dtype, tokens_list
            )

        if targets is None or "trajectory" not in targets:
            raise RuntimeError("Teacher SFT requires trajectory targets for the goal objective.")
        gt_traj = targets["trajectory"].cuda().to(model_dtype)

        rows, teacher_traj = self._lookup_rollouts(tokens_list, batch_size, training=True)
        if rows.numel() == 0:
            # Every rank must produce gradients or DDP will hang on the allreduce,
            # so an empty batch has to fail loudly rather than be skipped.
            raise RuntimeError(
                "No sample in this batch has a teacher rollout. Check that "
                "teacher_rollout_dir matches the caches this run trains on."
            )
        coverage = self._update_coverage(rows.numel(), batch_size)

        sel = rows.cuda()
        teacher_traj = teacher_traj.cuda().to(model_dtype)
        # The goal the DiT is conditioned on is the chosen rollout's OWN endpoint,
        # which is what makes the (goal, trajectory) pair self-consistent.
        sft_goal = teacher_traj[:, -1, :]

        vl_sel = last_hidden.index_select(0, sel)
        his_sel = history_flat.index_select(0, sel).to(model_dtype)
        status_sel = status.index_select(0, sel).to(model_dtype)
        state_sel = torch.cat([status_sel, his_sel], dim=1)

        action_inputs = BatchFeature(data={
            "state": state_sel,
            "his_traj": his_sel,
            "status_feature": status_sel,
            "action": teacher_traj,
            "goal": sft_goal,
        })
        # Identical epsilon-MSE to IL / teacher training; only the target
        # trajectory and the goal source differ.
        diff_out = planner(vl_sel, action_inputs)
        sft_loss = diff_out.loss

        # Goal head: detached inputs mean this touches goal_predictor only.
        pred_goal, pred_goal_norm = planner.predict_goal(vl_sel, his_sel, status_sel)
        gt_goal = gt_traj.index_select(0, sel)[:, -1, :]
        goal_target = sft_goal if self.goal_target == "teacher_canonical" else gt_goal
        goal_target_norm = planner.norm_odo(goal_target.unsqueeze(1)).squeeze(1).to(pred_goal_norm.dtype)
        # The conditioner ignores heading while goal_use_heading=False, so do not
        # spend goal-head capacity on a channel that cannot affect the plan.
        use_heading = bool(getattr(getattr(planner, "goal_encoder", None), "use_heading", False))
        goal_dims = 3 if use_heading else 2
        goal_loss = F.smooth_l1_loss(
            pred_goal_norm[..., :goal_dims], goal_target_norm[..., :goal_dims], reduction="mean"
        )

        total = sft_loss + self.goal_loss_weight * goal_loss

        with torch.no_grad():
            goal_fde = (pred_goal.float()[..., :2] - gt_goal.float()[..., :2]).norm(dim=-1).mean()
            teacher_fde = (teacher_traj.float()[:, -1, :2] - gt_goal.float()[..., :2]).norm(dim=-1).mean()
            # Acceptance metric: a zero norm means the goal branch never received
            # gradient and would still be a no-op when OPD starts.
            goal_encoder_wnorm = planner.goal_encoder.net[-1].weight.detach().float().norm()

        return BatchFeature(data={
            "loss": total,
            "sft_loss": sft_loss.detach(),
            "goal_loss": goal_loss.detach(),
            "weighted_goal_loss": (self.goal_loss_weight * goal_loss).detach(),
            "goal_fde_m": goal_fde,
            "teacher_fde_gt_m": teacher_fde,
            "rollout_coverage": torch.tensor(coverage, device=total.device),
            "goal_encoder_wnorm": goal_encoder_wnorm,
        })

    def _eval_forward(
        self, planner, last_hidden, history_flat, status, targets, model_dtype, tokens_list=None
    ):
        """Deployable validation path: plan on the student's own predicted goal."""
        status = status.to(model_dtype)
        history_flat = history_flat.to(model_dtype)
        action_inputs = BatchFeature(data={
            "state": torch.cat([status, history_flat], dim=1),
            "his_traj": history_flat,
            "status_feature": status,
        })
        pred_goal, _ = planner.predict_goal(last_hidden, history_flat, status)
        with planner.goal_context(pred_goal):
            out = planner.get_action(last_hidden, action_inputs)
        out["pred_goal"] = pred_goal.detach()

        # Acceptance metric #1: how close the deployable student now plans to the
        # routed teacher it is imitating. This is the curve that should flatten
        # before SFT is stopped -- driving it to zero overfits the static labels
        # and costs plasticity in the OPD stage.
        rows, teacher_traj = self._lookup_rollouts(tokens_list, last_hidden.shape[0], training=False)
        # Surfaced even when zero: the val scenes come from a different index than
        # the Phase A rollouts, so a 0% hit rate here is a real possibility and
        # would otherwise just make the acceptance metric quietly never appear.
        out["val_rollout_coverage"] = torch.tensor(
            rows.numel() / max(last_hidden.shape[0], 1), device=last_hidden.device
        )
        if rows.numel() > 0:
            sel = rows.to(out["pred_traj"].device)
            student_end = out["pred_traj"].float().index_select(0, sel)[:, -1, :2]
            teacher_end = teacher_traj.to(student_end.device).float()[:, -1, :2]
            out["student_fde_teacher_m"] = (student_end - teacher_end).norm(dim=-1).mean()

        self._eval_batches += 1
        if self.goal_sensitivity_interval > 0 and self._eval_batches % self.goal_sensitivity_interval == 0:
            out["goal_sensitivity_m"] = self._goal_sensitivity(
                planner, last_hidden, action_inputs, pred_goal, targets
            )
        return out

    @torch.no_grad()
    def _goal_sensitivity(self, planner, last_hidden, action_inputs, pred_goal, targets) -> torch.Tensor:
        """How far the plan moves when the goal is replaced by a wrong one.

        A non-zero ``goal_encoder`` weight norm only proves gradient flowed. This
        proves the goal actually participates in the decision: if swapping the
        goal barely moves the trajectory, the branch is decorative and the OPD
        stage has nothing to build on.
        """
        if targets is None or "trajectory" not in targets:
            return torch.zeros((), device=pred_goal.device)
        with planner.goal_context(pred_goal):
            base = planner.get_action(last_hidden, action_inputs)["pred_traj"].float()
        shuffled = pred_goal[torch.randperm(pred_goal.shape[0], device=pred_goal.device)]
        with planner.goal_context(shuffled):
            other = planner.get_action(last_hidden, action_inputs)["pred_traj"].float()
        return (base[:, -1, :2] - other[:, -1, :2]).norm(dim=-1).mean()

    # ------------------------------------------------------------------- losses
    def compute_loss(self, features, targets, predictions):
        if self.training:
            return predictions
        pred = torch.nan_to_num(predictions["pred_traj"], nan=0.0, posinf=0.0, neginf=0.0)
        tgt = torch.nan_to_num(targets["trajectory"], nan=0.0, posinf=0.0, neginf=0.0)
        tgt = tgt.to(device=pred.device, dtype=pred.dtype)
        loss = F.l1_loss(pred, tgt)
        data = {"loss": loss}
        for key in ("goal_sensitivity_m", "student_fde_teacher_m", "val_rollout_coverage"):
            if key in predictions:
                data[key] = predictions[key]
        if "pred_goal" in predictions:
            goal_fde = (
                predictions["pred_goal"].float()[..., :2] - tgt.float()[:, -1, :2]
            ).norm(dim=-1).mean()
            data["goal_fde_m"] = goal_fde
        return BatchFeature(data=data)

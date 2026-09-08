"""Lightning wrapper for goal-teacher scene-router OPD. Additive file.

Extends ``AgentLightningSceneRouter`` with:
* logging of the goal-distillation diagnostics emitted by
  ``ReCogDriveDiTSceneRouterGoalDistillTrainer`` (only ``loss`` uses
  ``sync_dist=True``; other scalars / per-bucket keys are rank-local to avoid
  NCCL SeqNum skew with DDP);
* periodic BEV visualisation (rank 0, ``on_train_batch_end``): GT trajectory +
  goal point, the routed teacher's final x0 and the student's final trajectory.
"""

import os
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module_scene_router import (
    AgentLightningSceneRouter,
)

# Global scalars: logged rank-locally (sync_dist=False); only loss is synced.
_GOAL_GLOBAL_KEYS = (
    "x0_gap_m",
    "x0_gap_final_m",
    "gauss_kl_mean",
    "entropy_mean",
    "student_fde_gt_m",
    "student_ade_gt_m",
    "teacher_goal_effect_m",
    # GoalBridge diagnostics
    "reverse_kl_loss",
    "goal_loss",
    "weighted_goal_loss",
    "anchor_loss",
    "weighted_anchor_loss",
    "goal_fde_m",
    "recoverability_mean",
    "recoverability_min",
    "recoverability_max",
    "pred_goal_x_mean",
    "pred_goal_y_mean",
)
# Data-dependent or per-step keys: rank-local logging (sync_dist=False).
_GOAL_LOCAL_PREFIXES = ("fde_gt_", "goal_effect_", "entropy_step_")


class AgentLightningSceneRouterGoal(AgentLightningSceneRouter):
    """Lightning wrapper for goal-teacher scene-router OPD with diagnostics."""

    def __init__(self, agent: AbstractAgent):
        super().__init__(agent)
        self.viz_interval_steps = int(getattr(agent, "viz_interval_steps", 0))

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], Any], logging_prefix: str) -> Tensor:
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features, targets, tokens_list)
        output = self.agent.compute_loss(features, targets, prediction)

        loss = output.loss if hasattr(output, "loss") else output
        # Only sync the training objective across ranks. Extra sync_dist=True
        # reductions interleaved with DDP backward have caused SeqNum skew /
        # ALLREDUCE timeouts on the first step of this job.
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if not isinstance(output, torch.Tensor):
            scalar_keys = (
                "distill_loss",
                "smooth_loss",
                "weighted_smooth_loss",
                "sigma_mean",
                "chain_abs_max",
                "denoising_steps",
                "exopd_lambda",
                "student_pred_traj_mean",
                "student_pred_traj_std",
            ) + _GOAL_GLOBAL_KEYS
            for key in scalar_keys:
                if key in output:
                    self.log(
                        f"{logging_prefix}/{key}",
                        output[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=key in ("distill_loss", "student_fde_gt_m"),
                        sync_dist=False,
                    )
            for key in list(output.keys()):
                if key.startswith(("kl_", "n_samples_") + _GOAL_LOCAL_PREFIXES):
                    self.log(
                        f"{logging_prefix}/{key}",
                        output[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=False,
                    )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # Rank-0 viz after the optimizer step so it cannot sit between
        # training_step logging collectives and DDP gradient allreduce.
        self._maybe_visualize()

    # ------------------------------------------------------------------- viz
    def _maybe_visualize(self) -> None:
        if self.viz_interval_steps <= 0 or self.global_rank != 0:
            return
        # Skip step 0: first-batch I/O + CUDA warmup is already the slowest path.
        if self.global_step <= 0 or self.global_step % self.viz_interval_steps != 0:
            return
        trainer_obj = getattr(self.agent, "scene_router_trainer", None)
        viz = getattr(trainer_obj, "last_viz", None)
        if not viz:
            return
        try:
            self._draw_bev(viz)
        except Exception as exc:  # viz must never kill training
            print(f"[SceneRouter-GoalOPD] viz skipped: {exc}")
        finally:
            trainer_obj.last_viz = None

    def _draw_bev(self, viz: Dict[str, Any]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt = viz["gt_traj"]            # (B, H, 3) metres
        goal = viz["goal"]             # (B, 3) GT endpoint
        pred_goal = viz.get("pred_goal")  # (B, 3) student Goal Head
        student = viz["student_traj"]  # (B, H, 3)
        teacher = viz["teacher_traj"]  # (B, H, 3)
        step_predictions = viz.get("step_predictions", [])
        buckets = viz["buckets"]
        # Light -> dark so early / mid / late chain states are separable.
        teacher_step_colors = ("#FFD27A", "#FF8C1A", "#9A3412")
        student_step_colors = ("#A5F3FC", "#22D3EE", "#0E7490")

        n = min(gt.shape[0], 8)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
        for i in range(rows * cols):
            ax = axes[i // cols][i % cols]
            if i >= n:
                ax.axis("off")
                continue
            # ego frame: x forward -> plot as vertical axis (BEV convention)
            # Draw chain diagnostics first so the solid final trajectories stay on top.
            for j, step_pred in enumerate(step_predictions):
                step = step_pred["step"]
                teacher_x0 = step_pred["teacher_x0"]
                student_x0 = step_pred["student_x0"]
                t_color = teacher_step_colors[min(j, len(teacher_step_colors) - 1)]
                s_color = student_step_colors[min(j, len(student_step_colors) - 1)]
                ax.plot(
                    -teacher_x0[i, :, 1],
                    teacher_x0[i, :, 0],
                    color=t_color,
                    linestyle="--",
                    alpha=1.0,
                    linewidth=1.6,
                    label=f"teacher x0 @ student z[{step}]" if i == 0 else None,
                )
                ax.plot(
                    -student_x0[i, :, 1],
                    student_x0[i, :, 0],
                    color=s_color,
                    linestyle=":",
                    alpha=1.0,
                    linewidth=1.6,
                    label=f"student x0 @ z[{step}]" if i == 0 else None,
                )
            ax.plot(-gt[i, :, 1], gt[i, :, 0], "k.-", label="GT", linewidth=1.5)
            ax.plot(-teacher[i, :, 1], teacher[i, :, 0], "r.-", label="teacher x0 (final z)", linewidth=1.8)
            ax.plot(-student[i, :, 1], student[i, :, 0], "b.-", label="student x0 (final z)", linewidth=1.8)
            ax.scatter([-goal[i, 1]], [goal[i, 0]], marker="*", s=180, c="g", label="GT goal", zorder=5)
            if pred_goal is not None:
                ax.scatter(
                    [-pred_goal[i, 1]],
                    [pred_goal[i, 0]],
                    marker="D",
                    s=55,
                    c="magenta",
                    edgecolors="white",
                    linewidths=0.6,
                    label="student pred goal" if i == 0 else None,
                    zorder=6,
                )
            ax.scatter([0.0], [0.0], marker="^", s=60, c="gray", zorder=5)
            ax.set_title(f"[{i}] {buckets[i]}", fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"goal-OPD step {self.global_step}", fontsize=11)
        fig.tight_layout()

        configured_out_dir = getattr(self.agent, "viz_output_dir", None)
        if configured_out_dir:
            out_dir = configured_out_dir
        else:
            log_dir = getattr(self.trainer, "log_dir", None) or "."
            out_dir = os.path.join(log_dir, "goal_opd_viz")
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"step_{self.global_step:08d}.png"), dpi=110)

        tb = getattr(getattr(self.trainer, "logger", None), "experiment", None)
        if tb is not None and hasattr(tb, "add_figure"):
            tb.add_figure("train/goal_opd_bev", fig, global_step=self.global_step)
        plt.close(fig)

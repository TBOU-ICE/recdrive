"""Lightning wrapper and goal-free BEV viz for scene-router KD-only OPD."""

import os
from typing import Any, Dict

import torch

from navsim.planning.training.agent_lightning_module_scene_router_goal import (
    AgentLightningSceneRouterGoal,
)


class AgentLightningSceneRouterKDOnly(AgentLightningSceneRouterGoal):
    """Reuse diagnostics logging and draw no-goal student/teacher comparisons."""

    def _draw_bev(self, viz: Dict[str, Any]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt = viz["gt_traj"]
        student = viz["student_traj"]
        teacher = viz["teacher_traj"]
        step_predictions = viz.get("step_predictions", [])
        buckets = viz["buckets"]
        if not step_predictions:
            step_predictions = [
                {
                    "step": "final",
                    "timestep": 0,
                    "student_x0": student,
                    "teacher_x0": teacher,
                }
            ]

        rows = min(gt.shape[0], 6)
        columns = len(step_predictions)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.2 * columns, 3.6 * rows),
            squeeze=False,
        )
        student_fde = torch.linalg.vector_norm(
            student[..., :2] - gt[..., :2], dim=-1
        )[:, -1]
        teacher_fde = torch.linalg.vector_norm(
            teacher[..., :2] - gt[..., :2], dim=-1
        )[:, -1]
        high_error_threshold_m = 2.0

        for row in range(rows):
            high_error = float(student_fde[row]) >= high_error_threshold_m
            for column, prediction in enumerate(step_predictions):
                axis = axes[row][column]
                teacher_x0 = prediction["teacher_x0"]
                student_x0 = prediction["student_x0"]
                step = prediction["step"]
                timestep = prediction.get("timestep", "?")
                axis.plot(
                    -gt[row, :, 1],
                    gt[row, :, 0],
                    "k.-",
                    label="GT",
                    linewidth=1.6,
                )
                axis.plot(
                    -teacher_x0[row, :, 1],
                    teacher_x0[row, :, 0],
                    "r.--",
                    label="teacher x0",
                    linewidth=1.8,
                )
                axis.plot(
                    -student_x0[row, :, 1],
                    student_x0[row, :, 0],
                    "b.:",
                    label="student x0",
                    linewidth=1.8,
                )
                axis.scatter([0.0], [0.0], marker="^", s=55, c="gray", zorder=5)
                if high_error:
                    axis.set_facecolor("#FFF1F2")
                axis.set_title(
                    f"student z[{step}] / DDIM t={timestep}", fontsize=9
                )
                axis.set_aspect("equal", adjustable="datalim")
                axis.grid(True, alpha=0.3)
                if column == 0:
                    status = "HIGH-ERR" if high_error else "OK"
                    axis.set_ylabel(
                        f"[{row}] {buckets[row]} | {status}\n"
                        f"S-FDE={float(student_fde[row]):.2f}m  "
                        f"T-FDE={float(teacher_fde[row]):.2f}m",
                        fontsize=8,
                    )
                if row == 0 and column == 0:
                    axis.legend(fontsize=7, loc="best")

        figure.suptitle(
            f"Scene-router KD-only matched x0 — step {self.global_step}\n"
            f"HIGH-ERR proxy: final student FDE >= {high_error_threshold_m:.1f}m",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))

        configured_output = getattr(self.agent, "viz_output_dir", None)
        if configured_output:
            output_dir = configured_output
        else:
            log_dir = getattr(self.trainer, "log_dir", None) or "."
            output_dir = os.path.join(log_dir, "scene_router_kd_only_viz")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"step_{self.global_step:08d}.png"
        )
        figure.savefig(output_path, dpi=110)

        tensorboard = getattr(
            getattr(self.trainer, "logger", None), "experiment", None
        )
        if tensorboard is not None and hasattr(tensorboard, "add_figure"):
            tensorboard.add_figure(
                "train/scene_router_kd_only_bev",
                figure,
                global_step=self.global_step,
            )
        plt.close(figure)

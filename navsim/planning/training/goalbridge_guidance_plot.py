"""Render the teacher-guidance diagnostic dumped by ``GoalBridgeGuidanceProbe``.

Kept separate from the probe so a plot can be restyled without re-running any
inference: the probe writes a ``.pt`` payload, this module reads it.

Two figures per payload:

``*_bev.png``
    Rows are DDIM steps, columns are samples, and every panel overlays the GT
    trajectory with the x0 predictions of each probed condition.  Step 0 is
    dropped by default: at ``sqrt(alpha_bar) ~ 0.31`` the x0 reconstruction
    amplifies any eps error roughly threefold before the clamp, so that row is
    ragged for every model and carries no signal about guidance quality.
    **All panels share one set of axis limits** -- per-panel autoscaling makes a
    perfectly ordinary trajectory look chaotic, which is exactly the misreading
    this diagnostic exists to avoid.

``*_summary.png``
    The conclusion figure.  Per-step ADE of each condition, plus the second
    moment of the student's z_t against the teacher's training marginal.  Read
    this before the BEV panels.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import torch

# Dark yellow rather than orange: at small linewidths orange reads as red.
CONDITION_STYLE: Dict[str, Dict[str, Any]] = {
    "student_pred": {
        "color": "#1F77B4",
        "label": "student x0 @ student z_t",
        "linewidth": 1.7,
        "alpha": 1.0,
        "zorder": 4,
        "linestyle": "-",
    },
    "teacher_gt": {
        "color": "#D62728",
        "label": "teacher x0 @ student z_t (GT goal)",
        "linewidth": 1.7,
        "alpha": 1.0,
        "zorder": 6,
        "linestyle": "-",
    },
    "teacher_pred": {
        "color": "#B8860B",
        "label": "teacher x0 @ student z_t (pred goal)",
        "linewidth": 1.2,
        "alpha": 0.75,
        "zorder": 3,
        "linestyle": "--",
    },
    "teacher_qgt": {
        "color": "#2CA02C",
        "label": "teacher x0 @ forward-noised GT (GT goal)",
        "linewidth": 1.2,
        "alpha": 0.75,
        "zorder": 3,
        "linestyle": "-.",
    },
}


def select_samples(payload: Dict[str, Any], n: int = 6) -> List[int]:
    """Stratify by the error that drives the hypothesis, not at random.

    With a trained goal head the stratifier is the goal error, since that is
    what recoverability gates on.  Goal-free payloads fall back to the final
    student FDE.
    """
    goal_err = payload["goal_err_m"]
    key = goal_err if torch.isfinite(goal_err).all() else payload["steps"][-1]["fde_m"]["student_pred"]
    order = torch.argsort(key)
    total = order.numel()
    if total <= n:
        return order.tolist()
    # Evenly spaced quantiles so low / mid / high error samples are all present.
    picks = torch.linspace(0, total - 1, steps=n).round().long()
    return order[picks].tolist()


def _shared_limits(payload, sample_idx, step_idx, conditions, pad: float = 2.0):
    xs, ys = [], []

    def _add(traj):
        xs.append(-traj[sample_idx][..., 1])
        ys.append(traj[sample_idx][..., 0])

    _add(payload["gt_traj"])
    for s in step_idx:
        for key in conditions:
            _add(payload["steps"][s]["traj"][key])
    x = torch.cat([v.reshape(-1) for v in xs])
    y = torch.cat([v.reshape(-1) for v in ys])
    return (
        (float(x.min()) - pad, float(x.max()) + pad),
        (float(y.min()) - pad, float(y.max()) + pad),
    )


def plot_bev(
    payload: Dict[str, Any],
    out_path: str,
    n_samples: int = 6,
    skip_steps: Sequence[int] = (0,),
    sample_idx: Optional[Sequence[int]] = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(payload["conditions"])
    steps = payload["steps"]
    step_idx = [i for i in range(len(steps)) if i not in set(skip_steps)]
    if sample_idx is None:
        sample_idx = select_samples(payload, n_samples)
    sample_idx = list(sample_idx)

    xlim, ylim = _shared_limits(payload, sample_idx, step_idx, conditions)

    # BEV keeps an equal aspect, so size each panel from the data aspect rather
    # than a fixed width. A square allotment around a tall, narrow corridor both
    # wastes the figure and shrinks the drawn frame away from the axes bbox,
    # which would leave the annotations floating outside the plot.
    n_rows, n_cols = len(step_idx), len(sample_idx)
    panel_h = 3.2
    data_aspect = (xlim[1] - xlim[0]) / max(ylim[1] - ylim[0], 1e-6)
    panel_w = min(max(panel_h * data_aspect, 1.6), 5.0) + 0.7
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False
    )

    gt = payload["gt_traj"]
    gt_goal = payload["gt_goal"]
    pred_goal = payload.get("pred_goal")

    for r, s in enumerate(step_idx):
        record = steps[s]
        for c, i in enumerate(sample_idx):
            ax = axes[r][c]
            ax.plot(
                -gt[i, :, 1], gt[i, :, 0],
                color="black", linestyle="--", linewidth=2.6, alpha=0.65,
                label="GT", zorder=2,
            )
            for key in conditions:
                style = CONDITION_STYLE[key]
                traj = record["traj"][key]
                ax.plot(
                    -traj[i, :, 1], traj[i, :, 0],
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["linewidth"], alpha=style["alpha"],
                    zorder=style["zorder"], label=style["label"],
                )
            ax.scatter([-gt_goal[i, 1]], [gt_goal[i, 0]], marker="*", s=130,
                       c="black", zorder=8, label="GT goal")
            if pred_goal is not None:
                ax.scatter([-pred_goal[i, 1]], [pred_goal[i, 0]], marker="*", s=130,
                           c="#1F77B4", edgecolors="white", linewidths=0.6,
                           zorder=9, label="pred goal")
            ax.scatter([0.0], [0.0], marker="^", s=45, c="gray", zorder=7)

            # Locked limits: autoscaling per panel is what makes a normal
            # trajectory look chaotic.
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

            # Saturation is the first thing to rule out before blaming guidance.
            clamp_bits = " ".join(
                f"{k.split('_')[0][0].upper()}{k.split('_')[-1][0]}="
                f"{float(record['clamp_frac'][k][i]):.2f}"
                for k in conditions
            )
            ax.text(
                0.02, 0.98, f"clamp {clamp_bits}",
                transform=ax.transAxes, fontsize=6, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
            if r == 0:
                goal_err = float(payload["goal_err_m"][i])
                goal_txt = "n/a" if goal_err != goal_err else f"{goal_err:.2f}m"
                ax.set_title(
                    f"[{i}] {payload['buckets'][i]}\ngoal err={goal_txt}", fontsize=8
                )
            if c == 0:
                ax.set_ylabel(
                    f"z[{record['step']}]  DDIM t={record['timestep']}\n"
                    f"sqrt(a_bar)={record['alpha_bar'] ** 0.5:.2f}  "
                    f"|z| ratio={record['z_second_moment_ratio']:.2f}",
                    fontsize=8,
                )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    meta = payload.get("meta", {})
    fig.suptitle(
        f"GoalBridge teacher-guidance probe — {meta.get('tag', '')}\n"
        f"student goal source = {meta.get('student_goal_source', '?')} | "
        f"ckpt = {os.path.basename(str(meta.get('student_ckpt', '')))} | "
        f"shared axis limits",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_summary(payload: Dict[str, Any], out_path: str) -> str:
    """Per-step error curves plus the off-distribution scalar. Read this first."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(payload["conditions"])
    steps = payload["steps"]
    x = [r["step"] for r in steps]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    for key in conditions:
        style = CONDITION_STYLE[key]
        axes[0].plot(
            x, [float(r["ade_m"][key].mean()) for r in steps],
            marker="o", color=style["color"], linestyle=style["linestyle"],
            label=style["label"],
        )
        axes[1].plot(
            x, [float(r["clamp_frac"][key].mean()) for r in steps],
            marker="o", color=style["color"], linestyle=style["linestyle"],
            label=style["label"],
        )
    axes[0].set_title("mean ||x0 - GT|| per DDIM step (m)")
    axes[0].set_xlabel("chain step (0 = noisiest)")
    axes[0].set_ylabel("ADE (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_title("x0 clamp saturation fraction")
    axes[1].set_xlabel("chain step")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)

    ratio = [r["z_second_moment_ratio"] for r in steps]
    axes[2].plot(x, ratio, marker="s", color="#444444", label="student z_t")
    axes[2].axhline(1.0, color="green", linestyle="--", linewidth=1.2,
                    label="teacher training marginal")
    axes[2].set_title("E[z_t^2] / (a*E[x0^2] + 1-a)")
    axes[2].set_xlabel("chain step")
    axes[2].set_ylabel("ratio")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    meta = payload.get("meta", {})
    fig.suptitle(
        f"GoalBridge guidance summary — {meta.get('tag', '')} "
        f"(student goal source = {meta.get('student_goal_source', '?')})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_payload(payload_path: str, out_dir: str, n_samples: int = 6,
                 skip_steps: Sequence[int] = (0,)) -> List[str]:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    tag = payload.get("meta", {}).get("tag", "probe")
    return [
        plot_summary(payload, os.path.join(out_dir, f"{tag}_summary.png")),
        plot_bev(payload, os.path.join(out_dir, f"{tag}_bev.png"),
                 n_samples=n_samples, skip_steps=skip_steps),
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", help="path to a .pt written by the diagnostic entry")
    parser.add_argument("--out-dir", default=None, help="default: alongside the payload")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument(
        "--skip-steps", default="0",
        help="comma-separated chain steps to omit from the BEV grid (default: 0)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.payload))
    skip = tuple(int(s) for s in args.skip_steps.split(",") if s.strip() != "")
    for path in plot_payload(args.payload, out_dir, args.n_samples, skip):
        print(f"[guidance-plot] wrote {path}")


if __name__ == "__main__":
    main()

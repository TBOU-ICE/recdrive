"""Render the teacher-guidance diagnostic dumped by ``GoalBridgeGuidanceProbe``.

Kept separate from the probe so a plot can be restyled without re-running any
inference: the probe writes a ``.pt`` payload, this module reads it.

Summary plus per-sample figures per payload:

``*_bev_by_error/{q1_low,q2_mid_low,q3_mid_high,q4_high}/*.png``
    One sample per image.  Four large panels show z1 through z4 left-to-right,
    with shared per-sample axis limits and dense metric ticks.  Every one of the
    balanced probe samples is saved and organized by error quartile.

``*_summary.png``
    The conclusion figure.  Per-step ADE of each condition, plus the second
    moment of the student's z_t against the teacher's training marginal.  Read
    this before the BEV panels.
"""

from __future__ import annotations

import os
import re
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


def _stratification_error(payload: Dict[str, Any]) -> tuple[torch.Tensor, str]:
    """Return the per-sample error used to form visualization strata."""
    goal_err = payload["goal_err_m"].float()
    if torch.isfinite(goal_err).all():
        return goal_err, "goal error"
    return payload["steps"][-1]["fde_m"]["student_pred"].float(), "final student FDE"


def _error_strata(payload: Dict[str, Any]):
    """Split every sample into four equally sized, error-sorted strata."""
    error, metric_name = _stratification_error(payload)
    order = torch.argsort(error)
    names = ("q1_low", "q2_mid_low", "q3_mid_high", "q4_high")
    chunks = torch.tensor_split(order, 4)
    return [
        (name, chunk.tolist(), error, metric_name)
        for name, chunk in zip(names, chunks)
    ]


def _safe_filename(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "no-token"))
    return text[:96]


def _plot_one_sample_bev(
    payload: Dict[str, Any],
    sample_index: int,
    step_idx: Sequence[int],
    out_path: str,
    stratum: str,
    error_value: float,
    metric_name: str,
) -> str:
    """One large figure per sample, with z1..z4 shown left-to-right."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    conditions = list(payload["conditions"])
    steps = payload["steps"]
    xlim, ylim = _shared_limits(
        payload, [sample_index], step_idx, conditions, pad=1.0
    )
    fig, axes = plt.subplots(
        1, len(step_idx), figsize=(5.8 * len(step_idx), 8.2), squeeze=False
    )
    axes = axes[0]
    gt = payload["gt_traj"]
    gt_goal = payload["gt_goal"]
    pred_goal = payload.get("pred_goal")

    for column, s in enumerate(step_idx):
        ax = axes[column]
        record = steps[s]
        ax.plot(
            -gt[sample_index, :, 1], gt[sample_index, :, 0],
            color="black", linestyle="--", linewidth=3.0, alpha=0.72,
            label="GT", zorder=2,
        )
        for key in conditions:
            style = CONDITION_STYLE[key]
            traj = record["traj"][key]
            ax.plot(
                -traj[sample_index, :, 1], traj[sample_index, :, 0],
                color=style["color"], linestyle=style["linestyle"],
                linewidth=max(2.0, style["linewidth"]), alpha=style["alpha"],
                zorder=style["zorder"], label=style["label"],
            )
        ax.scatter(
            [-gt_goal[sample_index, 1]], [gt_goal[sample_index, 0]],
            marker="*", s=190, c="black", zorder=8, label="GT goal",
        )
        if pred_goal is not None:
            ax.scatter(
                [-pred_goal[sample_index, 1]], [pred_goal[sample_index, 0]],
                marker="*", s=190, c="#1F77B4", edgecolors="white",
                linewidths=0.8, zorder=9, label="pred goal",
            )
        ax.scatter([0.0], [0.0], marker="^", s=70, c="gray", zorder=7)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        # Dense metric ticks make lateral/longitudinal deviations readable.
        ax.xaxis.set_major_locator(MultipleLocator(2.0))
        ax.yaxis.set_major_locator(MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        ax.grid(True, which="major", alpha=0.34, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.16, linewidth=0.5)
        ax.tick_params(labelsize=10)
        ax.set_xlabel("lateral (m), 2 m/tick", fontsize=11)
        if column == 0:
            ax.set_ylabel("longitudinal (m), 2 m/tick", fontsize=11)

        clamp_bits = "  ".join(
            f"{key}={float(record['clamp_frac'][key][sample_index]):.2f}"
            for key in conditions
        )
        ax.set_title(
            f"z{record['step']}   DDIM t={record['timestep']}\n"
            f"sqrt(a_bar)={record['alpha_bar'] ** 0.5:.3f}   "
            f"|z| ratio={record['z_second_moment_ratio']:.3f}\n"
            f"clamp: {clamp_bits}",
            fontsize=10,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=10, frameon=False
    )
    token = payload.get("tokens")
    token_text = token[sample_index] if token is not None else "n/a"
    meta = payload.get("meta", {})
    fig.suptitle(
        f"{meta.get('tag', '')} | sample={sample_index} | "
        f"bucket={payload['buckets'][sample_index]} | {stratum}\n"
        f"{metric_name}={error_value:.3f} m | token={token_text}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.91))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bev_by_error_strata(
    payload: Dict[str, Any],
    out_dir: str,
    step_numbers: Sequence[int] = (1, 2, 3, 4),
) -> List[str]:
    """Save every sample separately under four error-quantile directories."""
    step_to_index = {int(record["step"]): i for i, record in enumerate(payload["steps"])}
    missing = [step for step in step_numbers if step not in step_to_index]
    if missing:
        raise ValueError(
            f"Requested DDIM chain steps {missing}, available={sorted(step_to_index)}"
        )
    step_idx = [step_to_index[step] for step in step_numbers]

    written: List[str] = []
    tokens = payload.get("tokens")
    for stratum, sample_indices, error, metric_name in _error_strata(payload):
        stratum_dir = os.path.join(out_dir, stratum)
        for rank, sample_index in enumerate(sample_indices):
            token = tokens[sample_index] if tokens is not None else f"sample-{sample_index}"
            filename = (
                f"{rank:03d}_sample-{sample_index:04d}_"
                f"{_safe_filename(payload['buckets'][sample_index])}_"
                f"{_safe_filename(token)}.png"
            )
            written.append(
                _plot_one_sample_bev(
                    payload=payload,
                    sample_index=sample_index,
                    step_idx=step_idx,
                    out_path=os.path.join(stratum_dir, filename),
                    stratum=stratum,
                    error_value=float(error[sample_index]),
                    metric_name=metric_name,
                )
            )
    return written


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


def plot_payload(payload_path: str, out_dir: str) -> List[str]:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    tag = payload.get("meta", {}).get("tag", "probe")
    outputs = [plot_summary(payload, os.path.join(out_dir, f"{tag}_summary.png"))]
    outputs.extend(
        plot_bev_by_error_strata(
            payload, os.path.join(out_dir, f"{tag}_bev_by_error"),
            step_numbers=(1, 2, 3, 4),
        )
    )
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", help="path to a .pt written by the diagnostic entry")
    parser.add_argument("--out-dir", default=None, help="default: alongside the payload")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.payload))
    for path in plot_payload(args.payload, out_dir):
        print(f"[guidance-plot] wrote {path}")


if __name__ == "__main__":
    main()

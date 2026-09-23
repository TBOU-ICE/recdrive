"""Plot state-correction losses, gradient conflict, validation, and trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path):
    return json.loads(path.read_text())


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else math.nan


def train_metric(row, key, statistic="mean"):
    value = row.get("metrics", {}).get(key)
    if isinstance(value, dict):
        value = value.get(statistic)
    return finite(value)


def validation_metric(report, step, key, statistic="mean"):
    value = report.get(str(step), {}).get(key)
    if isinstance(value, dict):
        value = value.get(statistic)
    return finite(value)


def plot_finetune(output_dir: Path, cfg: dict) -> list[Path]:
    folder = output_dir / "finetune"
    # A completed finetune may have been produced on a data machine while
    # validation/plots are required on the workspace volume.  Read the source
    # logs from the explicit checkpoint location, but always write plots under
    # the configured output_dir.
    source_folder = folder
    checkpoint = cfg.get("finetuned_teacher_checkpoint")
    if not (source_folder / "train.jsonl").is_file() and checkpoint:
        source_folder = Path(checkpoint).expanduser().parent
    log_path = source_folder / "train.jsonl"
    if not log_path.is_file():
        return []
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not rows:
        return []

    steps = [row["global_step"] for row in rows]
    corr_weight = float(cfg["finetune"]["correction_weight"])
    retain_weight = float(cfg["finetune"]["retention_weight"])
    corr_loss = [train_metric(row, "correction_loss") for row in rows]
    retain_loss = [train_metric(row, "retention_eps_loss") for row in rows]

    fig, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(steps, [row["loss"] for row in rows], label="total")
    ax.plot(steps, [corr_weight * value for value in corr_loss], label="weighted correction")
    ax.plot(steps, [retain_weight * value for value in retain_loss], label="weighted retention")
    ax.set(title="Training objectives", xlabel="global step", ylabel="loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, [train_metric(row, "z4/ade_m") for row in rows], label="z4 ADE")
    ax.plot(steps, [train_metric(row, "z4/fde_m") for row in rows], label="z4 FDE")
    ax.plot(steps, [train_metric(row, "z4/ade_gain_m") for row in rows], label="ADE gain from state")
    ax.set(title="Rank-0 training batch geometry", xlabel="global step", ylabel="meters")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    for key, label in (
        ("branches/weighted_correction_grad_norm", "correction (all params)"),
        ("branches/weighted_retention_grad_norm", "retention (all params)"),
        ("decoder_branches/weighted_correction_grad_norm", "correction (decoder)"),
        ("decoder_branches/weighted_retention_grad_norm", "retention (decoder)"),
    ):
        ax.plot(steps, [train_metric(row, key) for row in rows], label=label)
    ax.set(title="Weighted branch gradient norms", xlabel="global step", ylabel="L2 norm")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(steps, [train_metric(row, "branches/gradient_cosine") for row in rows], label="all params")
    ax.plot(steps, [train_metric(row, "decoder_branches/gradient_cosine") for row in rows], label="decoder")
    ax.set(title="Correction vs retention gradient cosine", xlabel="global step", ylabel="cosine", ylim=(-1.05, 1.05))
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    ax.axhline(1.0, color="black", linewidth=1)
    ax.plot(steps, [train_metric(row, "branches/weighted_grad_norm_ratio") for row in rows], label="all params")
    ax.plot(steps, [train_metric(row, "decoder_branches/weighted_grad_norm_ratio") for row in rows], label="decoder")
    ax.set(title="Weighted correction / retention gradient ratio", xlabel="global step", ylabel="ratio")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    val_paths = sorted(source_folder.glob("val_epoch_*.json"))
    epochs = []
    val_ade = []
    val_fde = []
    val_qgt = []
    val_p90 = []
    terminal_step = max(cfg["collect"]["steps"])
    for path in val_paths:
        report = read_json(path)
        epochs.append(int(path.stem.rsplit("_", 1)[-1]))
        val_ade.append(validation_metric(report, terminal_step, "ade_m"))
        val_fde.append(validation_metric(report, terminal_step, "fde_m"))
        val_qgt.append(validation_metric(report, terminal_step, "qgt_ade_m"))
        val_p90.append(validation_metric(report, terminal_step, "ade_m", "p90"))
    if epochs:
        ax.plot(epochs, val_ade, marker="o", label="z4 ADE mean")
        ax.plot(epochs, val_p90, marker="o", label="z4 ADE P90")
        ax.plot(epochs, val_fde, marker="o", label="z4 FDE mean")
        ax.plot(epochs, val_qgt, marker="o", label="q(GT) ADE mean")
    ax.set(title="Validation by epoch", xlabel="epoch", ylabel="meters")
    ax.grid(alpha=0.25)
    ax.legend()

    plot_dir = folder / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output = plot_dir / "finetune_diagnostics.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return [output]


def trajectory_xy(row, key):
    trajectory = row.get(key)
    if trajectory is None:
        return None
    # BEV convention used by the repository: lateral=-y, longitudinal=x.
    lateral = [-point[1] for point in trajectory]
    longitudinal = [point[0] for point in trajectory]
    return lateral, longitudinal


def plot_paired_trajectories(output_dir: Path, count: int) -> list[Path]:
    folder = output_dir / "paired_validation"
    source = folder / "per_sample.json"
    if not source.is_file():
        return []
    rows = [
        row for row in read_json(source)
        if row.get("step", -1) >= 0 and "teacher_x0_trajectory" in row
        and "baseline_teacher_x0_trajectory" in row
    ]
    if not rows:
        return []
    rows.sort(key=lambda row: row.get("paired_ade_gain_m", 0.0))
    count = max(1, int(count))
    middle = max(0, len(rows) // 2 - count // 2)
    selected = (
        [("largest regressions", row) for row in rows[:count]]
        + [("median", row) for row in rows[middle:middle + count]]
        + [("largest improvements", row) for row in rows[-count:]]
    )

    fig, axes = plt.subplots(3, count, figsize=(5 * count, 14), squeeze=False, constrained_layout=True)
    styles = (
        ("gt_trajectory", "GT", "black", "-"),
        ("baseline_teacher_x0_trajectory", "original teacher", "tab:orange", "--"),
        ("teacher_x0_trajectory", "adapted teacher", "tab:blue", "-"),
        ("student_state_trajectory", "student z4", "0.55", ":"),
        ("student_final_trajectory", "student final", "tab:green", "-."),
    )
    for ax, (group, row) in zip(axes.flat, selected):
        for key, label, color, linestyle in styles:
            points = trajectory_xy(row, key)
            if points is not None:
                ax.plot(*points, label=label, color=color, linestyle=linestyle, marker="o", markersize=2)
        gain = row.get("paired_ade_gain_m", math.nan)
        ax.set_title(f"{group}\n{row['token']}  gain={gain:+.3f} m")
        ax.set_xlabel("lateral -y (m)")
        ax.set_ylabel("longitudinal x (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plot_dir = folder / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    trajectory_output = plot_dir / "paired_trajectory_viz.png"
    fig.savefig(trajectory_output, dpi=160)
    plt.close(fig)

    gains = [row.get("paired_ade_gain_m", math.nan) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.hist(gains, bins=60, color="tab:blue", alpha=0.8)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set(title="Paired ADE gain: original minus adapted", xlabel="ADE gain (m; positive is better)", ylabel="samples")
    ax.grid(alpha=0.25)
    gain_output = plot_dir / "paired_ade_gain_distribution.png"
    fig.savefig(gain_output, dpi=160)
    plt.close(fig)
    return [trajectory_output, gain_output]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory-count", type=int, default=3)
    args = parser.parse_args()
    cfg = read_json(Path(args.config))
    output_dir = Path(cfg["output_dir"])
    outputs = plot_finetune(output_dir, cfg)
    outputs.extend(plot_paired_trajectories(output_dir, args.trajectory_count))
    if outputs:
        for path in outputs:
            print(path)
    else:
        print("No finetune or paired-validation metrics are available yet.")


if __name__ == "__main__":
    main()

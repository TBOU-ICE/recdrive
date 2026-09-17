#!/usr/bin/env python3
"""Compare deterministic Phase-C probe outputs from base-init and SFT-init OPD.

The script refuses to compare runs whose effective dataset, probe, seed, world
size, batch size, source indexes, or recorded epoch sample order differ.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch


METRICS = (
    "student_teacher_ade_m",
    "student_teacher_fde_m",
    "student_gt_ade_m",
    "student_gt_fde_m",
    "teacher_gt_ade_m",
    "teacher_gt_fde_m",
    "goal_fde_m",
    "reverse_kl_loss",
    "x0_gap_m",
    "z1_x0_gap_m",
    "z3_x0_gap_m",
)
CURVE_METRICS = (
    "student_teacher_ade_m",
    "student_teacher_fde_m",
    "student_gt_ade_m",
    "student_gt_fde_m",
    "goal_fde_m",
    "reverse_kl_loss",
    "x0_gap_m",
    "z1_x0_gap_m",
    "z3_x0_gap_m",
)
METRIC_TITLES = {
    "student_teacher_ade_m": "Student–teacher ADE (m)",
    "student_teacher_fde_m": "Student–teacher FDE (m)",
    "student_gt_ade_m": "Student–GT ADE (m)",
    "student_gt_fde_m": "Student–GT FDE (m)",
    "teacher_gt_ade_m": "Teacher–GT ADE (m)",
    "teacher_gt_fde_m": "Teacher–GT FDE (m)",
    "goal_fde_m": "Predicted-goal endpoint error (m)",
    "reverse_kl_loss": "Shared-state reverse-KL surrogate",
    "x0_gap_m": "Mean student–teacher x0 gap (m)",
    "z1_x0_gap_m": "Student–teacher x0 gap at z1 (m)",
    "z3_x0_gap_m": "Student–teacher x0 gap at z3 (m)",
}
RUN_COLORS = {"base": "#E67E22", "sft": "#1976D2"}
RUN_LABELS = {"base": "No-SFT init", "sft": "SFT init"}
HALO = [path_effects.Stroke(linewidth=4.2, foreground="white"), path_effects.Normal()]


def _resolve_run(path: Path) -> Path:
    path = path.resolve()
    if (path / "run_manifest.json").is_file():
        return path
    candidates = sorted(path.rglob("run_manifest.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    if not candidates:
        raise FileNotFoundError(f"No run_manifest.json under {path}")
    raise RuntimeError(
        f"Found {len(candidates)} runs under {path}; pass one timestamped run directory"
    )


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_manifests(base_run: Path, sft_run: Path) -> Tuple[dict, dict]:
    base = _json(base_run / "run_manifest.json")
    sft = _json(sft_run / "run_manifest.json")
    required_equal = (
        "git_commit",
        "code_sha256",
        "dataset_sha256",
        "probe_sha256",
        "seed",
        "world_size",
        "batch_size_per_rank",
        "drop_last",
        "train_samples",
        "sampler_samples_per_epoch",
        "index_sha256",
        "experiment_config",
    )
    mismatches = {
        key: (base.get(key), sft.get(key))
        for key in required_equal
        if base.get(key) != sft.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Runs are not controlled comparisons; manifest mismatch:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    if base.get("init_mode") == sft.get("init_mode"):
        raise RuntimeError(
            f"Expected different initializations, both are {base.get('init_mode')!r}"
        )
    return base, sft


def _read_order_hashes(run: Path) -> Dict[int, str]:
    path = run / "epoch_order_sha256.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing sampler audit: {path}")
    rows: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        epoch = int(payload["epoch"])
        digest = str(payload["sha256"])
        if epoch in rows and rows[epoch] != digest:
            raise RuntimeError(f"Conflicting order hashes for epoch {epoch} in {path}")
        rows[epoch] = digest
    return rows


def _validate_orders(base_run: Path, sft_run: Path) -> List[int]:
    base = _read_order_hashes(base_run)
    sft = _read_order_hashes(sft_run)
    common = sorted(set(base) & set(sft))
    if not common:
        raise RuntimeError("The two runs have no completed audited epoch in common")
    bad = [epoch for epoch in common if base[epoch] != sft[epoch]]
    if bad:
        raise RuntimeError(f"Training sample order differs at epoch(s): {bad}")
    return common


def _available_steps(run: Path) -> List[int]:
    output = run / "probe_outputs"
    steps = []
    for path in output.glob("step_*"):
        try:
            step = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (path / "summary.json").is_file() and list(path.glob("rank_*.pt")):
            steps.append(step)
    return sorted(set(steps))


def _step_dir(run: Path, step: int) -> Path:
    return run / "probe_outputs" / f"step_{step:08d}"


def _probe_shard_dirs(run: Path) -> List[Path]:
    """Find DDP shard directories split by per-rank Hydra ``now`` resolution.

    Hydra resolves ``experiment_uid=${now:...}`` independently in every
    torchrun process.  When ranks cross a wall-clock second, rank outputs land
    in two adjacent timestamp directories.  The directory containing the run
    manifest is canonical; only manifest-less siblings within two seconds are
    accepted as shard-only companions, so unrelated reruns are never mixed.
    """
    directories = [run]
    try:
        run_time = datetime.strptime(run.name, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        return directories
    for sibling in run.parent.iterdir():
        if sibling == run or not sibling.is_dir():
            continue
        if (sibling / "run_manifest.json").exists():
            continue
        if not (sibling / "probe_outputs").is_dir():
            continue
        try:
            sibling_time = datetime.strptime(sibling.name, "%Y.%m.%d.%H.%M.%S")
        except ValueError:
            continue
        if abs((sibling_time - run_time).total_seconds()) <= 2.0:
            directories.append(sibling)
    return sorted(directories)


def _load_records(run: Path, step: int) -> Dict[str, dict]:
    files = sorted(
        path
        for shard_dir in _probe_shard_dirs(run)
        for path in _step_dir(shard_dir, step).glob("rank_*.pt")
    )
    if not files:
        raise FileNotFoundError(f"No rank outputs for step {step} under {run}")
    manifest = _json(run / "run_manifest.json")
    expected_ranks = int(manifest.get("world_size", 1))
    rank_names = {path.name for path in files}
    if len(rank_names) != expected_ranks:
        raise RuntimeError(
            f"Incomplete probe shards for {run} step {step}: "
            f"found {sorted(rank_names)}, expected {expected_ranks} ranks; "
            f"searched {[str(path) for path in _probe_shard_dirs(run)]}"
        )
    records: Dict[str, dict] = {}
    for path in files:
        rows = torch.load(path, map_location="cpu", weights_only=False)
        for row in rows:
            token = str(row["token"])
            if token in records:
                raise RuntimeError(f"Duplicate probe token {token} in {run} step {step}")
            records[token] = row
    return records


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _validate_records(base: Mapping[str, dict], sft: Mapping[str, dict]) -> None:
    if set(base) != set(sft):
        raise RuntimeError(
            f"Probe token mismatch: base-only={len(set(base)-set(sft))}, "
            f"sft-only={len(set(sft)-set(base))}"
        )
    for token in base:
        left, right = base[token], sft[token]
        for key in ("bucket", "noise_seed", "draw_viz"):
            if left[key] != right[key]:
                raise RuntimeError(f"Probe metadata mismatch token={token} key={key}")
        for key, atol in (
            ("gt_traj", 0.0),
            ("gt_goal", 0.0),
            ("teacher_traj", 1e-3),
            ("z1_teacher_x0", 1e-3),
            ("z3_teacher_x0", 1e-3),
        ):
            if not np.allclose(
                _as_numpy(left[key]), _as_numpy(right[key]), atol=atol, rtol=1e-5
            ):
                maximum = float(
                    np.max(np.abs(_as_numpy(left[key]) - _as_numpy(right[key])))
                )
                raise RuntimeError(
                    f"Common probe reference changed: token={token} key={key} "
                    f"max_abs_diff={maximum:.6g}"
                )


def _load_summaries(run: Path, steps: Sequence[int]) -> Dict[int, dict]:
    return {step: _json(_step_dir(run, step) / "summary.json") for step in steps}


def _write_summary_table(
    output: Path,
    base_summary: Mapping[int, dict],
    sft_summary: Mapping[int, dict],
) -> None:
    rows = []
    for run_name, summaries in (("base", base_summary), ("sft", sft_summary)):
        for step, payload in summaries.items():
            for bucket, values in (("overall", payload["overall"]), *payload["by_bucket"].items()):
                row = {"run": run_name, "step": step, "bucket": bucket}
                row.update({metric: float(values[metric]) for metric in METRICS})
                rows.append(row)
    with (output / "probe_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run", "step", "bucket", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    (output / "probe_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _plot_curves(
    output: Path,
    steps: Sequence[int],
    base_summary: Mapping[int, dict],
    sft_summary: Mapping[int, dict],
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), squeeze=False)
    for ax, metric in zip(axes.flat, CURVE_METRICS):
        for name, summaries in (("base", base_summary), ("sft", sft_summary)):
            values = [float(summaries[step]["overall"][metric]) for step in steps]
            ax.plot(
                steps,
                values,
                marker="o",
                markersize=4,
                linewidth=2,
                color=RUN_COLORS[name],
                label=RUN_LABELS[name],
            )
        ax.set_title(METRIC_TITLES[metric], fontsize=10)
        ax.set_xlabel("optimizer step")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=True, fontsize=9)
    fig.suptitle("Fixed 4×32 held-out probe: OPD evolution", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "metric_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bucket_bars(
    output: Path, step: int, base_summary: dict, sft_summary: dict
) -> None:
    metrics = (
        "student_teacher_ade_m",
        "student_teacher_fde_m",
        "student_gt_ade_m",
        "goal_fde_m",
        "reverse_kl_loss",
        "x0_gap_m",
    )
    buckets = sorted(base_summary["by_bucket"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    x = np.arange(len(buckets))
    width = 0.36
    short = [bucket.replace("_", "\n", 1) for bucket in buckets]
    for ax, metric in zip(axes.flat, metrics):
        base_values = [base_summary["by_bucket"][b][metric] for b in buckets]
        sft_values = [sft_summary["by_bucket"][b][metric] for b in buckets]
        ax.bar(x - width / 2, base_values, width, color=RUN_COLORS["base"], label=RUN_LABELS["base"])
        ax.bar(x + width / 2, sft_values, width, color=RUN_COLORS["sft"], label=RUN_LABELS["sft"])
        ax.set_xticks(x, short, fontsize=8)
        ax.set_title(METRIC_TITLES[metric], fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Fixed probe by scene class — optimizer step {step}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / f"bucket_comparison_step_{step:08d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_gap_delta_curves(
    output: Path,
    steps: Sequence[int],
    base_summary: Mapping[int, dict],
    sft_summary: Mapping[int, dict],
) -> None:
    """Plot SFT-minus-base directly; negative means SFT has the smaller gap."""
    panels = (
        ("Trajectory gap", ("student_teacher_ade_m", "student_teacher_fde_m")),
        ("Goal endpoint gap", ("goal_fde_m",)),
        ("Denoised x0 gap", ("x0_gap_m", "z1_x0_gap_m", "z3_x0_gap_m")),
        ("Reverse-transition gap", ("reverse_kl_loss",)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    colors = ("#6A1B9A", "#00897B", "#C62828")
    for ax, (title, metrics) in zip(axes.flat, panels):
        for color, metric in zip(colors, metrics):
            delta = [
                float(sft_summary[step]["overall"][metric])
                - float(base_summary[step]["overall"][metric])
                for step in steps
            ]
            ax.plot(steps, delta, marker="o", linewidth=2, color=color, label=METRIC_TITLES[metric])
        ax.axhline(0.0, color="black", linewidth=1.2, linestyle="--")
        ax.fill_between(steps, 0.0, ax.get_ylim()[0], color="#2E7D32", alpha=0.06)
        ax.set_title(f"{title}: SFT − No-SFT")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("delta (negative = SFT better)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("SFT advantage over the same fixed probe", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "gap_delta_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _paired_sample_stats(
    base_records: Mapping[str, dict], sft_records: Mapping[str, dict]
) -> Dict[str, dict]:
    metrics = (
        "student_teacher_ade_m",
        "student_teacher_fde_m",
        "student_gt_ade_m",
        "student_gt_fde_m",
        "goal_fde_m",
        "reverse_kl_loss",
        "x0_gap_m",
        "z1_x0_gap_m",
        "z3_x0_gap_m",
    )
    result = {}
    tokens = sorted(base_records)
    for metric in metrics:
        base = np.asarray([float(base_records[token][metric]) for token in tokens])
        sft = np.asarray([float(sft_records[token][metric]) for token in tokens])
        delta = sft - base
        result[metric] = {
            "num_samples": len(tokens),
            "base_mean": float(base.mean()),
            "sft_mean": float(sft.mean()),
            "mean_delta_sft_minus_base": float(delta.mean()),
            "median_delta_sft_minus_base": float(np.median(delta)),
            "sft_better_fraction": float((delta < 0).mean()),
            "tie_fraction": float(np.isclose(delta, 0.0, atol=1e-9).mean()),
            "delta_p10": float(np.quantile(delta, 0.10)),
            "delta_p90": float(np.quantile(delta, 0.90)),
        }
    return result


def _plot_paired_scatter(
    output: Path,
    step: int,
    base_records: Mapping[str, dict],
    sft_records: Mapping[str, dict],
) -> None:
    metrics = (
        "student_teacher_ade_m",
        "student_teacher_fde_m",
        "goal_fde_m",
        "x0_gap_m",
    )
    tokens = sorted(base_records)
    bucket_colors = {
        "general_or_no_tag": "#455A64",
        "progress_curbside_stopgo": "#1976D2",
        "rule_intersection": "#7B1FA2",
        "safety_dynamics_interaction": "#D84315",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        base = np.asarray([float(base_records[token][metric]) for token in tokens])
        sft = np.asarray([float(sft_records[token][metric]) for token in tokens])
        buckets = [str(base_records[token]["bucket"]) for token in tokens]
        for bucket in sorted(set(buckets)):
            mask = np.asarray([item == bucket for item in buckets])
            ax.scatter(
                base[mask], sft[mask], s=28, alpha=0.72,
                color=bucket_colors[bucket], label=bucket,
            )
        low = float(min(base.min(), sft.min()))
        high = float(max(base.max(), sft.max()))
        margin = max((high - low) * 0.04, 1e-8)
        ax.plot([low - margin, high + margin], [low - margin, high + margin], "k--", linewidth=1.2)
        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(low - margin, high + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("No-SFT")
        ax.set_ylabel("SFT")
        ax.set_title(METRIC_TITLES[metric] + "\n(points below diagonal: SFT better)")
        ax.grid(True, alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(f"Paired 128-sample comparison — optimizer step {step}", fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(output / f"paired_scatter_step_{step:08d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _xy(trajectory) -> Tuple[np.ndarray, np.ndarray]:
    data = _as_numpy(trajectory)
    return -data[..., 1], data[..., 0]


def _trajectory(
    ax,
    trajectory,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.2,
    alpha: float = 1.0,
    marker: str | None = None,
) -> None:
    x, y = _xy(trajectory)
    line = ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        marker=marker,
        markersize=3,
        zorder=3,
    )[0]
    line.set_path_effects(HALO)


def _setup_bev(ax, title: str) -> None:
    ax.scatter([0.0], [0.0], marker="^", s=75, color="#555555", zorder=8, label="ego")
    ax.axhline(0.0, color="#BBBBBB", linewidth=0.7, zorder=0)
    ax.axvline(0.0, color="#BBBBBB", linewidth=0.7, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("lateral −y (m)")
    ax.set_ylabel("forward x (m)")
    ax.set_title(title)


def _plot_token(
    output: Path,
    step: int,
    token: str,
    base: Mapping[str, object],
    sft: Mapping[str, object],
    residual_scale: float,
) -> None:
    fig = plt.figure(figsize=(18, 20))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.15, 1.0, 1.0), hspace=0.28, wspace=0.18)
    final_ax = fig.add_subplot(grid[0, :])
    z1_ax = fig.add_subplot(grid[1, 0])
    z3_ax = fig.add_subplot(grid[1, 1])
    residual_ax = fig.add_subplot(grid[2, 0])
    error_ax = fig.add_subplot(grid[2, 1])

    _trajectory(final_ax, base["gt_traj"], label="GT", color="#111111", linewidth=3.1)
    _trajectory(final_ax, base["teacher_traj"], label="Teacher (GT goal)", color="#D32F2F", linewidth=2.8)
    _trajectory(
        final_ax,
        base["student_traj"],
        label=RUN_LABELS["base"],
        color=RUN_COLORS["base"],
        linestyle="--",
        linewidth=2.7,
    )
    _trajectory(
        final_ax,
        sft["student_traj"],
        label=RUN_LABELS["sft"],
        color=RUN_COLORS["sft"],
        linestyle="-.",
        linewidth=2.7,
    )
    gt_goal = _as_numpy(base["gt_goal"])
    base_goal = _as_numpy(base["pred_goal"])
    sft_goal = _as_numpy(sft["pred_goal"])
    final_ax.scatter(-gt_goal[1], gt_goal[0], marker="*", s=260, color="#2E7D32", edgecolor="white", linewidth=1.1, label="GT goal", zorder=10)
    final_ax.scatter(-base_goal[1], base_goal[0], marker="D", s=95, color=RUN_COLORS["base"], edgecolor="white", linewidth=1.0, label="No-SFT predicted goal", zorder=10)
    final_ax.scatter(-sft_goal[1], sft_goal[0], marker="D", s=95, color=RUN_COLORS["sft"], edgecolor="white", linewidth=1.0, label="SFT predicted goal", zorder=10)
    _setup_bev(final_ax, "Final deterministic trajectories — true physical scale")
    final_ax.legend(ncol=4, loc="upper left", fontsize=9, framealpha=0.92)

    for ax, prefix, title in (
        (z1_ax, "z1", "Common-state denoising prediction at z1"),
        (z3_ax, "z3", "Common-state denoising prediction at z3"),
    ):
        _trajectory(ax, base["gt_traj"], label="GT", color="#111111", linewidth=2.2, alpha=0.65)
        _trajectory(ax, base[f"{prefix}_teacher_x0"], label="Teacher x0", color="#D32F2F", linewidth=2.6)
        _trajectory(ax, base[f"{prefix}_student_x0"], label="No-SFT student x0", color=RUN_COLORS["base"], linestyle="--")
        _trajectory(ax, sft[f"{prefix}_student_x0"], label="SFT student x0", color=RUN_COLORS["sft"], linestyle="-.")
        _setup_bev(ax, title)
        ax.legend(fontsize=8, loc="best")

    teacher = _as_numpy(base["teacher_traj"])
    base_student = _as_numpy(base["student_traj"])
    sft_student = _as_numpy(sft["student_traj"])
    base_amplified = teacher + residual_scale * (base_student - teacher)
    sft_amplified = teacher + residual_scale * (sft_student - teacher)
    _trajectory(residual_ax, teacher, label="Teacher reference", color="#D32F2F", linewidth=2.8)
    _trajectory(residual_ax, base_amplified, label=f"No-SFT residual ×{residual_scale:g}", color=RUN_COLORS["base"], linestyle="--")
    _trajectory(residual_ax, sft_amplified, label=f"SFT residual ×{residual_scale:g}", color=RUN_COLORS["sft"], linestyle="-.")
    _setup_bev(residual_ax, f"Residual amplification around teacher (NON-PHYSICAL, ×{residual_scale:g})")
    residual_ax.legend(fontsize=8, loc="best")

    gt = _as_numpy(base["gt_traj"])
    waypoints = np.arange(1, gt.shape[0] + 1)
    for name, student in (("base", base_student), ("sft", sft_student)):
        to_teacher = np.linalg.norm(student[:, :2] - teacher[:, :2], axis=-1)
        to_gt = np.linalg.norm(student[:, :2] - gt[:, :2], axis=-1)
        error_ax.plot(waypoints, to_teacher, color=RUN_COLORS[name], linewidth=2.5, marker="o", label=f"{RUN_LABELS[name]} → teacher")
        error_ax.plot(waypoints, to_gt, color=RUN_COLORS[name], linewidth=2.0, linestyle=":", marker="s", label=f"{RUN_LABELS[name]} → GT")
    error_ax.set_xlabel("future waypoint (0.5 s spacing)")
    error_ax.set_ylabel("XY distance (m)")
    error_ax.set_title("Per-waypoint error — true distances")
    error_ax.grid(True, alpha=0.25)
    error_ax.legend(fontsize=8, loc="best")
    metric_text = (
        "                 No-SFT      SFT\n"
        f"S–T ADE (m)   {float(base['student_teacher_ade_m']):8.3f}  {float(sft['student_teacher_ade_m']):8.3f}\n"
        f"S–T FDE (m)   {float(base['student_teacher_fde_m']):8.3f}  {float(sft['student_teacher_fde_m']):8.3f}\n"
        f"S–GT ADE (m)  {float(base['student_gt_ade_m']):8.3f}  {float(sft['student_gt_ade_m']):8.3f}\n"
        f"Goal FDE (m)  {float(base['goal_fde_m']):8.3f}  {float(sft['goal_fde_m']):8.3f}\n"
        f"Probe KL      {float(base['reverse_kl_loss']):8.4f}  {float(sft['reverse_kl_loss']):8.4f}"
    )
    error_ax.text(
        0.02,
        0.98,
        metric_text,
        transform=error_ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    fig.suptitle(
        f"{base['bucket']} | token {token} | optimizer step {step}",
        fontsize=16,
        y=0.995,
    )
    safe_token = token.replace("/", "_")
    fig.savefig(output / f"token_{safe_token}_step_{step:08d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _final_report(
    output: Path,
    step: int,
    steps: Sequence[int],
    epochs: Sequence[int],
    base_manifest: dict,
    sft_manifest: dict,
    base_summary: dict,
    sft_summary: dict,
) -> None:
    delta = {
        metric: float(sft_summary["overall"][metric])
        - float(base_summary["overall"][metric])
        for metric in METRICS
    }
    report = {
        "format": "student-to-teacher-opd-comparison-v1",
        "selected_step": step,
        "common_probe_steps": list(steps),
        "matched_order_epochs": list(epochs),
        "dataset_sha256": base_manifest["dataset_sha256"],
        "probe_sha256": base_manifest["probe_sha256"],
        "base": {
            "checkpoint": base_manifest["student_checkpoint"],
            "overall": base_summary["overall"],
        },
        "sft": {
            "checkpoint": sft_manifest["student_checkpoint"],
            "overall": sft_summary["overall"],
        },
        "sft_minus_base": delta,
        "lower_is_better_for_all_reported_metrics": True,
    }
    (output / "comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True, help="No-SFT/base-init Phase-C run")
    parser.add_argument("--sft-run", type=Path, required=True, help="SFT-init Phase-C run")
    parser.add_argument("--step", type=int, default=None, help="Common probe optimizer step; default latest")
    parser.add_argument("--residual-scale", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.residual_scale <= 0:
        raise ValueError("--residual-scale must be positive")

    base_run = _resolve_run(args.base_run)
    sft_run = _resolve_run(args.sft_run)
    base_manifest, sft_manifest = _validate_manifests(base_run, sft_run)
    common_epochs = _validate_orders(base_run, sft_run)
    common_steps = sorted(set(_available_steps(base_run)) & set(_available_steps(sft_run)))
    if not common_steps:
        raise RuntimeError("The two runs have no complete fixed-probe step in common")
    step = common_steps[-1] if args.step is None else int(args.step)
    if step not in common_steps:
        raise ValueError(f"step {step} is unavailable; common steps: {common_steps}")

    output = (args.output or (sft_run.parent / f"phase_c_compare_step_{step:08d}")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_summaries = _load_summaries(base_run, common_steps)
    sft_summaries = _load_summaries(sft_run, common_steps)
    _write_summary_table(output, base_summaries, sft_summaries)
    _plot_curves(output, common_steps, base_summaries, sft_summaries)
    _plot_gap_delta_curves(output, common_steps, base_summaries, sft_summaries)
    _plot_bucket_bars(output, step, base_summaries[step], sft_summaries[step])

    paired_steps = sorted({0, step})
    paired_stats = {}
    records_by_step = {}
    for paired_step in paired_steps:
        base_records_at_step = _load_records(base_run, paired_step)
        sft_records_at_step = _load_records(sft_run, paired_step)
        _validate_records(base_records_at_step, sft_records_at_step)
        records_by_step[paired_step] = (base_records_at_step, sft_records_at_step)
        paired_stats[str(paired_step)] = _paired_sample_stats(
            base_records_at_step, sft_records_at_step
        )
        _plot_paired_scatter(
            output, paired_step, base_records_at_step, sft_records_at_step
        )
    (output / "paired_sample_stats.json").write_text(
        json.dumps(paired_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    base_records, sft_records = records_by_step[step]
    viz_tokens = sorted(
        token for token, record in base_records.items() if bool(record["draw_viz"])
    )
    for token in viz_tokens:
        _plot_token(
            output,
            step,
            token,
            base_records[token],
            sft_records[token],
            float(args.residual_scale),
        )
    _final_report(
        output,
        step,
        common_steps,
        common_epochs,
        base_manifest,
        sft_manifest,
        base_summaries[step],
        sft_summaries[step],
    )
    print(f"comparison accepted: dataset/probe/config/order hashes match")
    print(f"selected step={step}; probe samples={len(base_records)}; detailed figures={len(viz_tokens)}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

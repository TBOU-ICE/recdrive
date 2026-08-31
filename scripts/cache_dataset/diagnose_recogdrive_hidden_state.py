#!/usr/bin/env python3
"""Diagnose ReCogDrive-VLM hidden-state caches: real navtrain vs SimScale.

Runs three checks (no ReCogDrive training):

1. Domain distribution: MMD, cosine similarity, PCA / t-SNE
2. Real-vs-sim binary classifier AUC (logistic regression on pooled hidden states)
   - AUC ~ 0.5: domains are close (desirable)
   - AUC > 0.9: large domain gap
3. Original (navtrain) vs perturbed (SimScale) hidden distance per source log
   - too small: VLM insensitive to SimScale perturbation
   - too large + far from real manifold: likely harmful for DiT

Example:
    PYTHONPATH=/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-scene \\
    python scripts/cache_dataset/diagnose_recogdrive_hidden_state.py \\
      --nav-cache /workspace/models/recdrive/v1.0.0/recogdrive_agent_cache_dir_train \\
      --sim-cache /workspace/datasets/simscale/20260709/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0 \\
      --output-dir /workspace/datasets/simscale/20260709/hidden_state_diag_round0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parents[1]

SIMSCALE_LOG_RE = re.compile(r"^(?P<base>.+)-(?P<hash>[0-9a-f]{16})-000$")


@dataclass
class SampleRecord:
    domain: str  # "navtrain" | "simscale"
    log_name: str
    token: str
    feature_path: Path
    base_log: str
    perturb_hash: Optional[str] = None


def _add_repo_to_path(repo_root: Path) -> None:
    repo = str(repo_root)
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)


def parse_simscale_log(log_name: str) -> Tuple[Optional[str], Optional[str]]:
    match = SIMSCALE_LOG_RE.match(log_name)
    if match is None:
        return None, None
    return match.group("base"), match.group("hash")


def pool_hidden_state(hidden: torch.Tensor, method: str = "mean") -> np.ndarray:
    """Convert (T, H) hidden states to a fixed-length vector."""
    arr = hidden.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 1:
        return arr
    if method == "mean":
        return arr.mean(axis=0)
    if method == "max":
        return arr.max(axis=0)
    if method == "cls":
        return arr[0]
    raise ValueError(f"Unknown pooling method: {method}")


def load_embedding(feature_path: Path, pool_method: str) -> np.ndarray:
    from navsim.planning.training.dataset import load_feature_target_from_pickle

    features = load_feature_target_from_pickle(feature_path)
    return pool_hidden_state(features["last_hidden_state"], method=pool_method)


def load_history(feature_path: Path) -> np.ndarray:
    from navsim.planning.training.dataset import load_feature_target_from_pickle

    features = load_feature_target_from_pickle(feature_path)
    return features["history_trajectory"].detach().cpu().numpy().astype(np.float32).reshape(-1)


def scan_cache(
    cache_path: Path,
    domain: str,
    max_records: int = 0,
    seed: int = 0,
) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    if not cache_path.is_dir():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

    log_dirs = [p for p in cache_path.iterdir() if p.is_dir()]
    if max_records > 0 and len(log_dirs) > max_records * 4:
        rng = random.Random(seed)
        rng.shuffle(log_dirs)

    for log_dir in sorted(log_dirs) if max_records <= 0 else log_dirs:
        log_name = log_dir.name
        if domain == "simscale":
            base_log, perturb_hash = parse_simscale_log(log_name)
        else:
            base_log, perturb_hash = log_name, None

        for token_dir in log_dir.iterdir():
            if not token_dir.is_dir():
                continue
            feature_path = token_dir / "internvl_feature.gz"
            if not feature_path.is_file():
                continue
            records.append(
                SampleRecord(
                    domain=domain,
                    log_name=log_name,
                    token=token_dir.name,
                    feature_path=feature_path,
                    base_log=base_log or log_name,
                    perturb_hash=perturb_hash,
                )
            )
            if max_records > 0 and len(records) >= max_records:
                return records
    return records


def scan_nav_for_base_logs(
    cache_path: Path,
    base_logs: Iterable[str],
    pool_method: str,
    max_anchors_per_log: int = 32,
    seed: int = 0,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, SampleRecord]]]:
    """Load navtrain hidden states for selected source logs (capped per log)."""
    index: Dict[str, List[Tuple[np.ndarray, np.ndarray, SampleRecord]]] = defaultdict(list)
    rng = random.Random(seed)
    wanted = set(base_logs)
    for base_log in sorted(wanted):
        log_dir = cache_path / base_log
        if not log_dir.is_dir():
            continue
        token_dirs = [p for p in log_dir.iterdir() if p.is_dir()]
        if max_anchors_per_log > 0 and len(token_dirs) > max_anchors_per_log:
            token_dirs = rng.sample(token_dirs, max_anchors_per_log)
        for token_dir in token_dirs:
            feature_path = token_dir / "internvl_feature.gz"
            if not feature_path.is_file():
                continue
            record = SampleRecord(
                domain="navtrain",
                log_name=base_log,
                token=token_dir.name,
                feature_path=feature_path,
                base_log=base_log,
            )
            try:
                emb = load_embedding(feature_path, pool_method=pool_method)
                hist = load_history(feature_path)
            except Exception:  # noqa: BLE001
                continue
            index[base_log].append((emb, hist, record))
    return index


def subsample(records: Sequence[SampleRecord], max_samples: int, seed: int) -> List[SampleRecord]:
    if max_samples <= 0 or len(records) <= max_samples:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(list(records), max_samples)


def build_embeddings(
    records: Sequence[SampleRecord],
    pool_method: str,
) -> Tuple[np.ndarray, List[SampleRecord]]:
    embeddings: List[np.ndarray] = []
    kept: List[SampleRecord] = []
    for idx, record in enumerate(records):
        try:
            emb = load_embedding(record.feature_path, pool_method=pool_method)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skip {record.feature_path}: {exc}")
            continue
        if not np.isfinite(emb).all():
            print(f"[WARN] non-finite embedding: {record.feature_path}")
            continue
        embeddings.append(emb)
        kept.append(record)
        if (idx + 1) % 500 == 0:
            print(f"  loaded {idx + 1}/{len(records)} embeddings")
    if not embeddings:
        raise RuntimeError("No embeddings loaded.")
    return np.stack(embeddings, axis=0), kept


def rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Unbiased-style RBF MMD estimate."""
    if gamma is None:
        combined = np.vstack([x, y])
        pairwise = np.sum((combined[None, :, :] - combined[:, None, :]) ** 2, axis=-1)
        gamma = 1.0 / max(float(np.median(pairwise[pairwise > 0])), 1e-6)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * np.sum(diff * diff, axis=-1))

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    m, n = len(x), len(y)
    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(mmd)


def cosine_stats(a: np.ndarray, b: np.ndarray, max_pairs: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}

    def _norm(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    a_n = _norm(a)
    b_n = _norm(b)
    pairs = min(max_pairs, len(a_n) * len(b_n))
    idx_a = rng.integers(0, len(a_n), size=pairs)
    idx_b = rng.integers(0, len(b_n), size=pairs)
    sims = np.sum(a_n[idx_a] * b_n[idx_b], axis=1)
    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "median": float(np.median(sims)),
        "p10": float(np.percentile(sims, 10)),
        "p90": float(np.percentile(sims, 90)),
    }


def train_domain_classifier(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
) -> Dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    pred = (prob >= 0.5).astype(np.int64)
    acc = float((pred == y_test).mean())
    return {"auc": float(auc), "accuracy": acc, "train_size": len(x_train), "test_size": len(x_test)}


def plot_domain_scatter(
    nav_emb: np.ndarray,
    sim_emb: np.ndarray,
    output_dir: Path,
    seed: int,
    max_points: int,
) -> None:
    rng = np.random.default_rng(seed)
    nav_idx = rng.choice(len(nav_emb), size=min(max_points, len(nav_emb)), replace=False)
    sim_idx = rng.choice(len(sim_emb), size=min(max_points, len(sim_emb)), replace=False)
    subset = np.vstack([nav_emb[nav_idx], sim_emb[sim_idx]])
    labels = np.array([0] * len(nav_idx) + [1] * len(sim_idx))

    pca = PCA(n_components=2, random_state=seed)
    pca_2d = pca.fit_transform(subset)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    tsne_2d = tsne.fit_transform(subset)

    for name, coords in [("pca", pca_2d), ("tsne", tsne_2d)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[labels == 0, 0], coords[labels == 0, 1], s=8, alpha=0.5, label="navtrain")
        ax.scatter(coords[labels == 1, 0], coords[labels == 1, 1], s=8, alpha=0.5, label="simscale")
        ax.legend()
        ax.set_title(f"Hidden-state {name.upper()} (pooled)")
        fig.tight_layout()
        fig.savefig(output_dir / f"domain_{name}.png", dpi=160)
        plt.close(fig)


def plot_histograms(output_dir: Path, values: Dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, arr in values.items():
        ax.hist(arr, bins=60, alpha=0.55, density=True, label=label)
    ax.legend()
    ax.set_title("Hidden-state distance / cosine distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "distance_histograms.png", dpi=160)
    plt.close(fig)


def nearest_nav_neighbor(
    nav_items: Sequence[Tuple[np.ndarray, np.ndarray, SampleRecord]],
    sim_history: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    if not nav_items:
        return None, float("nan")
    best_dist = float("inf")
    best_emb: Optional[np.ndarray] = None
    for emb, hist, _ in nav_items:
        dist = float(np.linalg.norm(hist - sim_history))
        if dist < best_dist:
            best_dist = dist
            best_emb = emb
    return best_emb, best_dist


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(1.0 - np.dot(a, b) / denom)


def analyze_original_vs_perturbed(
    sim_records: Sequence[SampleRecord],
    nav_index: Dict[str, List[Tuple[np.ndarray, np.ndarray, SampleRecord]]],
    nav_centroid: np.ndarray,
    pool_method: str,
    max_pairs: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rng = random.Random(seed)
    eligible = [r for r in sim_records if r.base_log in nav_index]
    if max_pairs > 0 and len(eligible) > max_pairs:
        eligible = rng.sample(eligible, max_pairs)

    rows: List[Dict[str, object]] = []
    l2_dists: List[float] = []
    cos_dists: List[float] = []
    nav_distances: List[float] = []
    intra_perturb_dists: List[float] = []

    grouped: Dict[str, List[Tuple[np.ndarray, SampleRecord]]] = defaultdict(list)
    for record in eligible:
        try:
            sim_emb = load_embedding(record.feature_path, pool_method=pool_method)
            sim_hist = load_history(record.feature_path)
        except Exception:  # noqa: BLE001
            continue

        nav_emb, hist_match_dist = nearest_nav_neighbor(nav_index[record.base_log], sim_hist)
        if nav_emb is None:
            continue

        l2 = l2_distance(nav_emb, sim_emb)
        cos = cosine_distance(nav_emb, sim_emb)
        nav_dist = l2_distance(sim_emb, nav_centroid)

        rows.append(
            {
                "base_log": record.base_log,
                "sim_log": record.log_name,
                "sim_token": record.token,
                "perturb_hash": record.perturb_hash or "",
                "history_match_l2": hist_match_dist,
                "original_vs_perturbed_l2": l2,
                "original_vs_perturbed_cosine": cos,
                "perturbed_to_nav_centroid_l2": nav_dist,
            }
        )
        l2_dists.append(l2)
        cos_dists.append(cos)
        nav_distances.append(nav_dist)
        grouped[record.base_log].append((sim_emb, record))

    for _, items in grouped.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                intra_perturb_dists.append(l2_distance(items[i][0], items[j][0]))

    def _summary(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    summary = {
        "pair_count": len(rows),
        "original_vs_perturbed_l2": _summary(l2_dists),
        "original_vs_perturbed_cosine": _summary(cos_dists),
        "perturbed_to_nav_centroid_l2": _summary(nav_distances),
        "intra_log_perturbation_l2": _summary(intra_perturb_dists),
    }

    if l2_dists:
        l2_arr = np.asarray(l2_dists)
        cos_arr = np.asarray(cos_dists)
        nav_arr = np.asarray(nav_distances)
        l2_p10, l2_p90 = np.percentile(l2_arr, [10, 90])
        nav_p90 = np.percentile(nav_arr, 90)
        too_small = int(np.sum(l2_arr <= l2_p10))
        too_large_off_manifold = int(np.sum((l2_arr >= l2_p90) & (nav_arr >= nav_p90)))
        summary["interpretation"] = {
            "too_small_count": too_small,
            "too_small_fraction": float(too_small / len(l2_arr)),
            "too_large_off_manifold_count": too_large_off_manifold,
            "too_large_off_manifold_fraction": float(too_large_off_manifold / len(l2_arr)),
            "note": (
                "too_small: original-vs-perturbed L2 <= p10; "
                "too_large_off_manifold: L2 >= p90 and perturbed far from nav centroid (>= p90)"
            ),
        }

    return rows, summary


def save_pair_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ReCogDrive hidden-state caches.")
    parser.add_argument(
        "--nav-cache",
        type=Path,
        default=Path("/workspace/models/recdrive/v1.0.0/recogdrive_agent_cache_dir_train"),
    )
    parser.add_argument(
        "--sim-cache",
        type=Path,
        default=Path(
            "/workspace/datasets/simscale/20260709/"
            "recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/datasets/simscale/20260709/hidden_state_diag"),
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--pool-method", choices=["mean", "max", "cls"], default="mean")
    parser.add_argument("--max-nav-samples", type=int, default=4000)
    parser.add_argument("--max-sim-samples", type=int, default=4000)
    parser.add_argument("--max-pairs", type=int, default=3000, help="Max original-vs-perturbed pairs.")
    parser.add_argument(
        "--max-nav-anchors-per-log",
        type=int,
        default=32,
        help="Navtrain anchors sampled per source log for perturbation matching.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_repo_to_path(args.repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning caches...")
    nav_records = scan_cache(
        args.nav_cache, domain="navtrain", max_records=args.max_nav_samples, seed=args.seed
    )
    sim_records = scan_cache(
        args.sim_cache, domain="simscale", max_records=args.max_sim_samples, seed=args.seed + 1
    )
    print(f"navtrain entries: {len(nav_records)}")
    print(f"simscale entries: {len(sim_records)}")

    print("Loading navtrain embeddings...")
    nav_emb, nav_kept = build_embeddings(nav_records, pool_method=args.pool_method)
    print("Loading simscale embeddings...")
    sim_emb, sim_kept = build_embeddings(sim_records, pool_method=args.pool_method)

    labels = np.array([0] * len(nav_emb) + [1] * len(sim_emb))
    all_emb = np.vstack([nav_emb, sim_emb])

    print("Computing domain metrics...")
    mmd = rbf_mmd(nav_emb, sim_emb)
    cosine_cross = cosine_stats(nav_emb, sim_emb, max_pairs=10000, seed=args.seed)
    cosine_nav = cosine_stats(nav_emb, nav_emb, max_pairs=5000, seed=args.seed + 2)
    cosine_sim = cosine_stats(sim_emb, sim_emb, max_pairs=5000, seed=args.seed + 3)
    classifier = train_domain_classifier(all_emb, labels, seed=args.seed, test_size=args.test_size)

    plot_domain_scatter(
        nav_emb,
        sim_emb,
        output_dir=args.output_dir,
        seed=args.seed,
        max_points=min(2000, max(len(nav_emb), len(sim_emb))),
    )

    print("Building nav index for original-vs-perturbed pairing...")
    pair_sim_records = sim_kept
    if args.max_pairs > 0 and len(pair_sim_records) > args.max_pairs:
        pair_sim_records = subsample(pair_sim_records, args.max_pairs, seed=args.seed + 4)
    base_logs = sorted({r.base_log for r in pair_sim_records})
    nav_index = scan_nav_for_base_logs(
        args.nav_cache,
        base_logs,
        pool_method=args.pool_method,
        max_anchors_per_log=args.max_nav_anchors_per_log,
        seed=args.seed + 5,
    )
    print(f"Loaded nav anchors for {len(nav_index)}/{len(base_logs)} base logs")
    nav_centroid = nav_emb.mean(axis=0)

    pair_rows, pair_summary = analyze_original_vs_perturbed(
        sim_records=pair_sim_records,
        nav_index=nav_index,
        nav_centroid=nav_centroid,
        pool_method=args.pool_method,
        max_pairs=0,
        seed=args.seed,
    )
    save_pair_csv(pair_rows, args.output_dir / "original_vs_perturbed_pairs.csv")

    # Re-render histogram with correct output dir (avoid passing None in helper)
    if pair_rows:
        plot_histograms(
            args.output_dir,
            {
                "orig_vs_pert_l2": np.array([r["original_vs_perturbed_l2"] for r in pair_rows]),
                "orig_vs_pert_cosine": np.array([r["original_vs_perturbed_cosine"] for r in pair_rows]),
                "pert_to_nav_centroid_l2": np.array([r["perturbed_to_nav_centroid_l2"] for r in pair_rows]),
            },
        )

    summary = {
        "nav_cache": str(args.nav_cache),
        "sim_cache": str(args.sim_cache),
        "pool_method": args.pool_method,
        "sample_counts": {
            "navtrain": len(nav_emb),
            "simscale": len(sim_emb),
        },
        "domain_metrics": {
            "mmd_rbf": mmd,
            "cosine_cross_domain": cosine_cross,
            "cosine_within_navtrain": cosine_nav,
            "cosine_within_simscale": cosine_sim,
            "binary_classifier": classifier,
        },
        "original_vs_perturbed": pair_summary,
        "guidance": {
            "binary_auc_near_0.5": "sim hidden close to real hidden (good)",
            "binary_auc_above_0.9": "large sim-real domain gap (risky)",
            "orig_vs_pert_too_small": "VLM may be insensitive to SimScale perturbation",
            "orig_vs_pert_too_large_off_manifold": "perturbed hidden may hurt DiT",
        },
    }

    summary_path = args.output_dir / "hidden_state_diagnostic_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Hidden-state diagnostic summary ===")
    print(f"MMD (RBF): {mmd:.6f}")
    print(f"Cross-domain cosine: mean={cosine_cross['mean']:.4f}, median={cosine_cross['median']:.4f}")
    print(f"Domain classifier AUC: {classifier['auc']:.4f}  accuracy={classifier['accuracy']:.4f}")
    if pair_summary.get("original_vs_perturbed_l2"):
        l2 = pair_summary["original_vs_perturbed_l2"]
        print(
            "Original vs perturbed L2: "
            f"median={l2['median']:.4f}, p10={l2['p10']:.4f}, p90={l2['p90']:.4f}"
        )
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

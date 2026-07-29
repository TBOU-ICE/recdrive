"""A1 for navhard at the official metric's unit: the two-stage scene group.

The saved eval CSV holds per-token scores, but the reported EPDMS is
``mean_over_groups( (s1(now) * wavg(s2_first) + s1(prev) * wavg(s2_second)) / 2 )``
(see ``calculate_individual_mapping_scores`` in navsim_v2's run_pdm_score).

Stage-two weights come from a Gaussian kernel over endpoint/start-point distances
that the CSV does not carry, so they are approximated as uniform here. The script
prints the reproduction error against the official log value so the approximation
can be judged.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

EXP_ROOT = Path("/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp")
SPLIT_DIR = Path("/workspace/navsim_v2/navsim/planning/script/config/common/train_test_split")
BUCKET_MAP = Path(
    "/workspace/volumes/ad-e2e-al-sh01/nby/data/navhard_scene/output/navhard/exclusive_token_to_bucket.json"
)

EXPERTS = {
    "progress": "eval-navhard-new-vlm-rl-expert-14epoch-progress-all-scenes",
    "rule": "eval-navhard-new-vlm-rl-expert-14epoch-rule-all-scenes",
    "safety": "eval-navhard-new-vlm-rl-expert-14epoch-safety-all-scenes",
    "general": "eval-navhard-new-vlm-rl-expert-11epoch-general-all-scenes",
}
EXPERT_BUCKET = {
    "progress": "progress_curbside_stopgo",
    "rule": "rule_intersection",
    "safety": "safety_dynamics_interaction",
    "general": "general_or_no_tag",
}
BUCKETS = list(EXPERT_BUCKET.values())
SHORT = {v: k for k, v in EXPERT_BUCKET.items()}


def normalize_token(token: str) -> str:
    token = str(token).strip().lower()
    base, sep, suffix = token.rpartition("-")
    if sep and suffix.isdigit() and len(suffix) == 3 and base:
        return token
    return token.replace("-", "")


def load_mapping() -> List[Tuple[str, str, List[Tuple[str, str]]]]:
    raw = yaml.safe_load((SPLIT_DIR / "navhard_two_stage.yaml").open())["reactive_all_mapping"]
    out = []
    for now, prev, pairs in raw:
        out.append(
            (
                normalize_token(now),
                normalize_token(prev),
                [(normalize_token(a), normalize_token(b)) for a, b in pairs],
            )
        )
    return out


def load_scores(eval_dir: str) -> Dict[str, float]:
    csvs = sorted(glob.glob(str(EXP_ROOT / eval_dir / "**" / "*.csv"), recursive=True))
    if not csvs:
        raise FileNotFoundError(f"no csv under {eval_dir}")
    df = pd.read_csv(csvs[-1], usecols=["token", "score"])
    df["token"] = df["token"].map(normalize_token)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).drop_duplicates("token")
    return dict(zip(df["token"], df["score"]))


def official_score(eval_dir: str) -> float:
    logs = glob.glob(str(EXP_ROOT / eval_dir / "**" / "*.log"), recursive=True)
    for log in logs:
        text = Path(log).read_text(errors="ignore")
        hit = re.search(r"Final extended pdm score of valid results: (\d+\.\d+)", text)
        if hit:
            return float(hit.group(1))
    return float("nan")


def group_scores(scores: Dict[str, float], mapping) -> np.ndarray:
    """Per-group EPDMS, replicating the official stage1 * stage2 product."""
    out = []
    for now, prev, pairs in mapping:
        first = [scores[a] for a, _ in pairs if a in scores]
        second = [scores[b] for _, b in pairs if b in scores]
        s1_now = scores.get(now, np.nan)
        s1_prev = scores.get(prev, np.nan)
        g1 = s1_now * float(np.mean(first)) if first else np.nan
        g2 = s1_prev * float(np.mean(second)) if second else np.nan
        out.append(np.nanmean([g1, g2]))
    return np.asarray(out, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(EXP_ROOT / "a1_disagreement"))
    args = parser.parse_args()

    mapping = load_mapping()
    raw_map = json.load(BUCKET_MAP.open())
    orig_bucket = {normalize_token(k): v for k, v in raw_map.items()}

    group_bucket = np.array([orig_bucket.get(now, "unmapped") for now, _, _ in mapping])
    expected = {}
    for bucket in BUCKETS:
        split = yaml.safe_load((SPLIT_DIR / f"navhard_{bucket}_two_stage.yaml").open())
        expected[bucket] = len(split["reactive_all_mapping"])
    got = {b: int((group_bucket == b).sum()) for b in BUCKETS}
    print(f"group counts by 'now' token bucket: {got}")
    print(f"group counts in per-bucket split yamls: {expected}")
    names = list(EXPERTS)

    print(f"navhard groups={len(mapping)}  bucket coverage={(group_bucket != 'unmapped').sum()}/{len(mapping)}")
    print()
    print("=== reproduction check (uniform stage-2 weights vs official log) ===")
    per_expert: Dict[str, np.ndarray] = {}
    for name in names:
        gs = group_scores(load_scores(EXPERTS[name]), mapping)
        per_expert[name] = gs
        repro, official = float(np.nanmean(gs)), official_score(EXPERTS[name])
        print(f"  {name:<9} repro={repro:.4f}  official={official:.4f}  diff={repro - official:+.4f}")

    mat = np.stack([per_expert[n] for n in names], axis=1)
    n_groups = len(mapping)

    print()
    print("=== Q1: scene x expert, mean group EPDMS ===")
    header = ["scene\\expert"] + names + ["n_groups", "best"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for bucket in BUCKETS:
        mask = group_bucket == bucket
        sub = mat[mask]
        means = [float(np.nanmean(sub[:, i])) for i in range(len(names))]
        cells = [SHORT[bucket]] + [f"{m:.4f}" for m in means] + [str(int(mask.sum())), names[int(np.argmax(means))]]
        print("| " + " | ".join(cells) + " |")

    print()
    print("=== Q2: per-group disagreement (max-min across experts) ===")
    spread = np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1)
    print(f"mean_spread={np.nanmean(spread):.4f}  median={np.nanmedian(spread):.4f}")
    for thr in (0.05, 0.1, 0.2):
        print(f"frac(spread>{thr})={np.nanmean(spread > thr) * 100:.1f}%")
    for bucket in BUCKETS:
        mask = group_bucket == bucket
        print(f"  {SHORT[bucket]}: {np.nanmean(spread[mask]):.4f} (n={int(mask.sum())})")

    print()
    print("=== Q3: strategy bounds (group-level) ===")
    col = {n: i for i, n in enumerate(names)}
    route_col = np.array([col[SHORT[b]] if b in SHORT else col["general"] for b in group_bucket])
    route = mat[np.arange(n_groups), route_col]
    oracle = np.nanmax(mat, axis=1)
    avg = np.nanmean(mat, axis=1)
    per_overall = {n: float(np.nanmean(mat[:, i])) for i, n in enumerate(names)}
    best_name = max(per_overall, key=per_overall.get)
    for name, value in per_overall.items():
        print(f"  {name}: {value:.4f}")
    print(f"best_single ({best_name}): {per_overall[best_name]:.4f}")
    print(f"average_of_experts: {np.nanmean(avg):.4f}")
    print(f"scene_route: {np.nanmean(route):.4f}")
    print(f"oracle_best_per_group: {np.nanmean(oracle):.4f}")
    print(f"oracle - scene_route: {np.nanmean(oracle) - np.nanmean(route):+.4f}")
    print(f"scene_route - best_single: {np.nanmean(route) - per_overall[best_name]:+.4f}")

    # Bootstrap over groups: is scene_route better than the best single expert?
    rng = np.random.default_rng(0)
    diffs = []
    best_idx = col[best_name]
    for _ in range(5000):
        idx = rng.integers(0, n_groups, n_groups)
        diffs.append(np.nanmean(route[idx]) - np.nanmean(mat[idx, best_idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"bootstrap 95% CI of (scene_route - best_single): [{lo:+.4f}, {hi:+.4f}]")

    os.makedirs(args.out, exist_ok=True)
    summary = {
        "bench": "navhard",
        "unit": "two_stage_scene_group",
        "n_groups": n_groups,
        "per_expert_overall": {k: round(v, 4) for k, v in per_overall.items()},
        "matrix": {
            SHORT[b]: {
                **{n: round(float(np.nanmean(mat[group_bucket == b, i])), 4) for i, n in enumerate(names)},
                "n_groups": int((group_bucket == b).sum()),
            }
            for b in BUCKETS
        },
        "strategy": {
            "average_of_experts": round(float(np.nanmean(avg)), 4),
            "scene_route": round(float(np.nanmean(route)), 4),
            "oracle": round(float(np.nanmean(oracle)), 4),
            "best_single": round(per_overall[best_name], 4),
            "best_single_name": best_name,
            "route_minus_best_single_ci95": [round(float(lo), 4), round(float(hi), 4)],
        },
    }
    out_path = Path(args.out) / "a1_navhard_group_level_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

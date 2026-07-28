#!/usr/bin/env python3
"""
A1: expert-disagreement analysis on cross-eval EPDMS CSVs (read-only, add-only).

For each bench (navhard / navtest), joins the 4 scenario experts' per-token EPDMS
(`score`) on the same token set and answers:
  (1) Is averaging harmful?  -> per-token score spread (max-min) distribution.
  (2) Is each expert best on its own scenario? -> per-bucket 4x4 mean-score matrix
      + own-expert win-rate (argmax == bucket's expert).
  (3) How much do routing / best-expert selection gain?
      avg(mean-of-experts)  <=  scene_route  <=  oracle(best per token)
      -> quantifies v1(route) gain over averaging and v2(pdm_best) headroom over route.

Usage:
  python analyze_expert_disagreement_a1.py [--bench navhard|navtest|both]
                                           [--spread-thresholds 0.05,0.1,0.2]
                                           [--out-dir <dir>]
"""

import argparse
import glob
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

EXP_ROOT = "/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
DATA_ROOT = "/workspace/volumes/ad-e2e-al-sh01/nby/data"

# expert short-name -> scenario bucket
EXPERT_BUCKET = {
    "progress": "progress_curbside_stopgo",
    "rule": "rule_intersection",
    "safety": "safety_dynamics_interaction",
    "general": "general_or_no_tag",
}
BUCKETS = list(EXPERT_BUCKET.values())
FALLBACK_BUCKET = "general_or_no_tag"

BENCHES = {
    "navhard": {
        "eval_dirs": {
            "general": "eval-navhard-new-vlm-rl-expert-11epoch-general-all-scenes",
            "progress": "eval-navhard-new-vlm-rl-expert-14epoch-progress-all-scenes",
            "rule": "eval-navhard-new-vlm-rl-expert-14epoch-rule-all-scenes",
            "safety": "eval-navhard-new-vlm-rl-expert-14epoch-safety-all-scenes",
        },
        # navhard tags live in per-bucket original+synthetic token lists (the single
        # exclusive_token_to_bucket.json only covers ~450 tokens), so build from parts.
        "bucket_parts_dir": f"{DATA_ROOT}/navhard_scene/output/navhard",
    },
    "navtest": {
        "eval_dirs": {
            "general": "eval-navsim2-new-vlm-rl-expert-11epoch-general-all-scenes",
            "progress": "eval-navsim2-new-vlm-rl-expert-14epoch-progress-all-scenes",
            "rule": "eval-navsim2-new-vlm-rl-expert-14epoch-rule-all-scenes",
            "safety": "eval-navsim2-new-vlm-rl-expert-14epoch-safety-all-scenes",
        },
        "bucket_json": f"{DATA_ROOT}/navtrain_scene/output/navtest/exclusive_token_to_bucket.json",
    },
}


def normalize_token(token) -> Optional[str]:
    if token is None:
        return None
    token = str(token).strip().lower()
    base, sep, suffix = token.rpartition("-")
    if not (sep and suffix.isdigit() and len(suffix) == 3 and base):
        token = token.replace("-", "")
    return token or None


def find_csv(eval_dir: str) -> str:
    hits = sorted(glob.glob(os.path.join(EXP_ROOT, eval_dir, "**", "*.csv"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No CSV under {eval_dir}")
    return hits[-1]


def load_expert_scores(eval_dir: str, name: str) -> pd.DataFrame:
    path = find_csv(eval_dir)
    df = pd.read_csv(path, usecols=lambda c: c in ("token", "score"))
    df = df[["token", "score"]].copy()
    df["token"] = df["token"].map(normalize_token)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["token", "score"]).drop_duplicates("token")
    return df.rename(columns={"score": name})


def load_bucket_map(json_path: str) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.items() if isinstance(raw, dict) else []
    mapping = {}
    for token, bucket in items:
        norm = normalize_token(token)
        if norm and isinstance(bucket, str):
            mapping[norm] = bucket if bucket in BUCKETS else FALLBACK_BUCKET
    return mapping


def load_bucket_map_from_parts(parts_dir: str) -> Dict[str, str]:
    """Build token->bucket from per-bucket exclusive_{bucket}_{original,synthetic}_tokens.json lists."""
    mapping: Dict[str, str] = {}
    for bucket in BUCKETS:
        for suffix in ("original", "synthetic"):
            p = os.path.join(parts_dir, f"exclusive_{bucket}_{suffix}_tokens.json")
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                tokens = json.load(f)
            for t in tokens:
                norm = normalize_token(t)
                if norm:
                    mapping[norm] = bucket
    return mapping


def analyze(bench: str, spread_thresholds, out_dir: Optional[str]) -> dict:
    cfg = BENCHES[bench]
    names = ["progress", "rule", "safety", "general"]
    dfs = [load_expert_scores(cfg["eval_dirs"][n], n) for n in names]
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="token", how="inner")
    n = len(merged)

    if "bucket_json" in cfg:
        bucket_map = load_bucket_map(cfg["bucket_json"])
    else:
        bucket_map = load_bucket_map_from_parts(cfg["bucket_parts_dir"])
    merged["bucket"] = merged["token"].map(bucket_map).fillna(FALLBACK_BUCKET)

    score_cols = names
    scores = merged[score_cols].to_numpy(dtype=float)  # [N,4]

    # (3) average vs scene_route vs oracle
    mean_of_experts = scores.mean(axis=1)              # optimistic proxy for "average teachers"
    oracle = scores.max(axis=1)                        # best-per-token (v2 pdm_best ceiling)
    worst = scores.min(axis=1)
    col_index = {nme: i for i, nme in enumerate(names)}
    bucket_to_expertcol = {b: col_index[e] for e, b in EXPERT_BUCKET.items()}
    route_col = merged["bucket"].map(bucket_to_expertcol).to_numpy()
    route = scores[np.arange(n), route_col]            # scene-routed expert's score

    per_expert_mean = {nme: float(scores[:, i].mean()) for i, nme in enumerate(names)}

    # (1) disagreement
    spread = scores.max(axis=1) - scores.min(axis=1)
    spread_frac = {f">{t}": float((spread > t).mean()) for t in spread_thresholds}

    # (2) per-bucket 4x4 mean matrix + own-expert win-rate
    argmax_expert = np.array(names)[scores.argmax(axis=1)]
    per_bucket = {}
    for b in BUCKETS:
        mask = (merged["bucket"] == b).to_numpy()
        cnt = int(mask.sum())
        if cnt == 0:
            per_bucket[b] = {"count": 0}
            continue
        sub = scores[mask]
        expert_means = {nme: float(sub[:, i].mean()) for i, nme in enumerate(names)}
        own_expert = [e for e, bk in EXPERT_BUCKET.items() if bk == b][0]
        own_col = col_index[own_expert]
        win_rate_own = float((scores[mask].argmax(axis=1) == own_col).mean())
        # which expert wins most often on this bucket
        vals, counts = np.unique(argmax_expert[mask], return_counts=True)
        top_winner = str(vals[counts.argmax()])
        best_mean_expert = max(expert_means, key=expert_means.get)
        per_bucket[b] = {
            "count": cnt,
            "expert_means": {k: round(v, 4) for k, v in expert_means.items()},
            "own_expert": own_expert,
            "own_expert_best_by_mean": bool(best_mean_expert == own_expert),
            "best_mean_expert": best_mean_expert,
            "own_expert_argmax_win_rate": round(win_rate_own, 4),
            "top_argmax_winner": top_winner,
        }

    result = {
        "bench": bench,
        "n_tokens_joined": n,
        "per_expert_overall_mean_score": {k: round(v, 4) for k, v in per_expert_mean.items()},
        "strategy_mean_score": {
            "worst_per_token": round(float(worst.mean()), 4),
            "average_of_experts": round(float(mean_of_experts.mean()), 4),
            "scene_route": round(float(route.mean()), 4),
            "oracle_best_per_token": round(float(oracle.mean()), 4),
        },
        "gains": {
            "route_minus_average": round(float(route.mean() - mean_of_experts.mean()), 4),
            "oracle_minus_route": round(float(oracle.mean() - route.mean()), 4),
            "oracle_minus_best_single_expert": round(
                float(oracle.mean() - max(per_expert_mean.values())), 4
            ),
        },
        "disagreement": {
            "mean_spread": round(float(spread.mean()), 4),
            "median_spread": round(float(np.median(spread)), 4),
            "frac_tokens_spread": {k: round(v, 4) for k, v in spread_frac.items()},
        },
        "per_bucket": per_bucket,
        "bucket_token_counts": {b: int((merged["bucket"] == b).sum()) for b in BUCKETS},
        "unmapped_tokens": int((~merged["token"].isin(bucket_map)).sum()),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"a1_{bench}_summary.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        merged.assign(
            mean_of_experts=mean_of_experts, scene_route=route, oracle=oracle, spread=spread
        ).to_csv(os.path.join(out_dir, f"a1_{bench}_per_token.csv"), index=False)
    return result


def print_report(res: dict) -> None:
    b = res["bench"]
    print(f"\n{'='*72}\n[A1] bench={b}  joined_tokens={res['n_tokens_joined']}  "
          f"unmapped={res['unmapped_tokens']}\n{'='*72}")
    print("per-expert overall EPDMS:", res["per_expert_overall_mean_score"])
    print("bucket token counts:", res["bucket_token_counts"])
    s = res["strategy_mean_score"]
    print(f"\n[3] strategy mean EPDMS (higher=better):")
    print(f"    average_of_experts = {s['average_of_experts']}   (>= real 'average teachers')")
    print(f"    scene_route        = {s['scene_route']}")
    print(f"    oracle_best/token  = {s['oracle_best_per_token']}")
    print(f"    gains: route-avg = {res['gains']['route_minus_average']:+}, "
          f"oracle-route(v2 headroom) = {res['gains']['oracle_minus_route']:+}, "
          f"oracle-best_single = {res['gains']['oracle_minus_best_single_expert']:+}")
    d = res["disagreement"]
    print(f"\n[1] disagreement: mean_spread={d['mean_spread']} median={d['median_spread']} "
          f"frac={d['frac_tokens_spread']}")
    print(f"\n[2] per-bucket (own-expert best-by-mean? / argmax win-rate):")
    for bk, info in res["per_bucket"].items():
        if info.get("count", 0) == 0:
            print(f"    {bk:32s} count=0")
            continue
        print(f"    {bk:32s} n={info['count']:5d}  own={info['own_expert']:8s} "
              f"own_best_by_mean={str(info['own_expert_best_by_mean']):5s} "
              f"best_mean={info['best_mean_expert']:8s} "
              f"own_argmax_win={info['own_expert_argmax_win_rate']}  "
              f"means={info['expert_means']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["navhard", "navtest", "both"], default="both")
    ap.add_argument("--spread-thresholds", default="0.05,0.1,0.2")
    ap.add_argument("--out-dir", default="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/a1_disagreement")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.spread_thresholds.split(",") if x.strip()]
    benches = ["navhard", "navtest"] if args.bench == "both" else [args.bench]
    for bench in benches:
        res = analyze(bench, thresholds, args.out_dir)
        print_report(res)
    print(f"\n[A1] summaries written to {args.out_dir}")


if __name__ == "__main__":
    main()

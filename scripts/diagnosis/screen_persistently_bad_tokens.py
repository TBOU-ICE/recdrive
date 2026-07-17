#!/usr/bin/env python3
"""Screen persistently bad / good tokens across multi-run PDMS CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_CSVS = {
    "fuxian": "/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/eval_recdrive/recdrive-fuxian-906/2026.04.21.19.01.41.csv",
    "il99": "/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/eval_recdrive_v2/eval-pdms-train-dit_il_fullmix_simscale_round01_quality-99epoch/2026.07.13.05.08.48/2026.07.13.06.39.56.csv",
    "rl_50_40": "/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/eval_recdrive_v2/eval-pdms-rule_rl_10nav_50navbucket_40simbucket-14epoch/2026.07.13.07.54.14/2026.07.13.08.23.06.csv",
    "rl_45_45": "/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/eval_recdrive_v2/eval-pdms-rule_rl_10nav_45navbucket_45simbucket-4epoch/2026.07.14.02.21.53/2026.07.14.02.46.49.csv",
    "rl_40_30": "/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/eval_recdrive_v2/eval-pdms-rule_rl_40nav_30navbucket_30simbucket-13epoch/2026.07.12.08.38.19/2026.07.12.09.02.50.csv",
}

RULE_KEYS = ["rl_50_40", "rl_45_45", "rl_40_30"]
ALL_KEYS = ["fuxian", "il99", "rl_50_40", "rl_45_45", "rl_40_30"]
CORE_KEYS = ["fuxian", "il99", "rl_45_45", "rl_40_30"]


def load_csv(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        tok = str(r["token"]).strip()
        out[tok] = {
            "score": float(r["score"]),
            "nc": float(r["no_at_fault_collisions"]),
            "dac": float(r["drivable_area_compliance"]),
        }
    return out


def percentile(vals: List[float], q: float) -> float:
    s = sorted(vals)
    return s[int(round((len(s) - 1) * q))]


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fail-use-n", type=int, default=150)
    p.add_argument("--good-use-n", type=int, default=150)
    p.add_argument("--good-min-score", type=float, default=0.95)
    p.add_argument("--hard-mean-core", type=float, default=0.5)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = {k: load_csv(v) for k, v in DEFAULT_CSVS.items()}
    public = set(runs[RULE_KEYS[0]])
    for k in RULE_KEYS[1:]:
        public &= set(runs[k])

    p20 = {k: percentile([runs[k][t]["score"] for t in public], 0.20) for k in ALL_KEYS}

    fail: List[dict] = []
    good: List[dict] = []
    for t in public:
        scores = {k: runs[k][t]["score"] for k in ALL_KEYS}
        safety = [k for k in ALL_KEYS if runs[k][t]["nc"] == 0.0 or runs[k][t]["dac"] == 0.0]
        mean_core = sum(scores[k] for k in CORE_KEYS) / len(CORE_KEYS)
        low_all = all(scores[k] < p20[k] for k in ALL_KEYS)
        hard = mean_core < args.hard_mean_core or len(safety) >= 2
        row = {
            "token": t,
            **{f"score_{k}": scores[k] for k in ALL_KEYS},
            "mean_core": mean_core,
            "safety_fail_runs": ",".join(safety),
            "n_safety_fail": len(safety),
            "low_all": low_all,
            "hard": hard,
        }
        if low_all or hard:
            fail.append(row)
        if all(scores[k] >= args.good_min_score for k in ALL_KEYS):
            good.append({k: row[k] for k in row if k not in ("low_all", "hard")})

    fail.sort(key=lambda r: (r["mean_core"], -r["n_safety_fail"]))
    good.sort(key=lambda r: -r["mean_core"])
    fail_hard = [r for r in fail if r["hard"]]
    fail_rest = [r for r in fail if not r["hard"]]
    fail_use = (fail_hard + fail_rest)[: args.fail_use_n]
    good_use = good[: args.good_use_n]

    (args.output_dir / "fail_tokens.txt").write_text("\n".join(r["token"] for r in fail_use) + "\n")
    (args.output_dir / "good_tokens.txt").write_text("\n".join(r["token"] for r in good_use) + "\n")
    (args.output_dir / "fail_tokens_all.txt").write_text("\n".join(r["token"] for r in fail) + "\n")
    (args.output_dir / "good_tokens_all.txt").write_text("\n".join(r["token"] for r in good) + "\n")
    write_csv(args.output_dir / "fail_token_scores.csv", fail_use)
    write_csv(args.output_dir / "good_token_scores.csv", good_use)
    write_csv(args.output_dir / "fail_token_scores_all.csv", fail)
    write_csv(args.output_dir / "good_token_scores_all.csv", good)

    summary = {
        "public_n": len(public),
        "p20": p20,
        "fail_n": len(fail),
        "good_n": len(good),
        "fail_use_n": len(fail_use),
        "good_use_n": len(good_use),
        "fail_hard_n": len(fail_hard),
        "criteria": {
            "fail": "score < per-run P20 on ALL CSVs, OR mean_core < hard_mean_core, OR safety fail in >=2 runs",
            "good": f"score >= {args.good_min_score} on ALL CSVs",
            "core_runs": CORE_KEYS,
        },
        "worst15": fail_use[:15],
    }
    (args.output_dir / "screen_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["public_n", "fail_n", "good_n", "fail_use_n", "good_use_n", "fail_hard_n"]}, indent=2))
    print(f"Wrote lists under {args.output_dir}")


if __name__ == "__main__":
    main()

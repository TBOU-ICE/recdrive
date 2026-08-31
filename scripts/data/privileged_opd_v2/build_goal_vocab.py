#!/usr/bin/env python3
"""Build an endpoint vocabulary from cached trajectory targets without sklearn.

Example:
  python scripts/data/privileged_opd_v2/build_goal_vocab.py \
    --cache-root /.../recogdrive_agent_cache_dir_train \
    --manifest data/epdms/manifests/nav_train_newvlm.json \
    --k 2048 --max-samples 100000 --out /.../goal_vocab_2048.npz
"""
import argparse, gzip, json, pickle
from pathlib import Path
import numpy as np


def norm(t):
    t = str(t).strip().lower(); base, sep, sfx = t.rpartition("-")
    return t if sep and sfx.isdigit() and len(sfx) == 3 and base else t.replace("-", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--token-json", action="append", default=[])
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--max-samples", type=int, default=100000)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    manifest = json.load(open(args.manifest))
    root = Path(args.cache_root)
    allowed = None
    for p in args.token_json:
        raw = json.load(open(p)); keys = raw.keys() if isinstance(raw, dict) else raw
        s = {norm(x) for x in keys}; allowed = s if allowed is None else allowed | s
    items = [(t, rel) for t, rel in manifest["tokens"].items() if allowed is None or norm(t) in allowed]
    rng.shuffle(items); items = items[:args.max_samples]

    endpoints = []
    for _, rel in items:
        p = root / rel / "trajectory_target.gz"
        try:
            with gzip.open(p, "rb") as f:
                traj = pickle.load(f)["trajectory"]
            a = np.asarray(traj, dtype=np.float32)
            endpoints.append(a[-1, :3])
        except Exception:
            continue
    x = np.asarray(endpoints, dtype=np.float32)
    if len(x) < args.k:
        raise RuntimeError(f"only {len(x)} valid endpoints for k={args.k}")
    xy = x[:, :2]
    centers = xy[rng.choice(len(xy), size=args.k, replace=False)].copy()
    assign = np.zeros(len(xy), dtype=np.int32)
    for it in range(args.iters):
        # chunked squared distances to keep RAM bounded
        for lo in range(0, len(xy), 4096):
            z = xy[lo:lo+4096]
            d = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            assign[lo:lo+len(z)] = d.argmin(1)
        counts = np.bincount(assign, minlength=args.k).astype(np.float32)
        sums = np.zeros_like(centers)
        np.add.at(sums, assign, xy)
        good = counts > 0
        centers[good] = sums[good] / counts[good, None]
        # reseed empty clusters from data
        empty = np.where(~good)[0]
        if len(empty): centers[empty] = xy[rng.choice(len(xy), len(empty), replace=False)]
        print(f"iter {it+1}/{args.iters}: mean_count={counts.mean():.1f}, empty={len(empty)}")

    # circular-mean heading per endpoint cluster
    sin_sum = np.zeros(args.k, np.float64); cos_sum = np.zeros(args.k, np.float64)
    np.add.at(sin_sum, assign, np.sin(x[:, 2])); np.add.at(cos_sum, assign, np.cos(x[:, 2]))
    heading = np.arctan2(sin_sum, cos_sum).astype(np.float32)
    goals = np.concatenate([centers.astype(np.float32), heading[:, None]], axis=1)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, goals=goals, counts=np.bincount(assign, minlength=args.k), source_n=len(x))
    print(f"saved {len(goals)} goals from {len(x)} endpoints -> {out}")

if __name__ == "__main__":
    main()

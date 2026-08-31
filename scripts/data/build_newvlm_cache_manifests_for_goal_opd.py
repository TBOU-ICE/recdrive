"""Build manifests for caches used by run_recogdrive_train_scene_router_dit_goal_distill_8gpu.sh.

One-time offline index so CacheOnlyDataset can skip multi-hour Alluxio walks.

Outputs (under data/epdms/manifests/):
  - nav_train_newvlm.json
  - sim_round0_quality_newvlm.json
  - sim_round1_quality_newvlm.json

Run:
  /mnt/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/python \\
    scripts/data/build_newvlm_cache_manifests_for_goal_opd.py
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT_DIR = Path("/mnt/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1/data/epdms/manifests")

# Matches the goal-distill 8gpu launch script defaults.
CACHES = [
    (
        Path(
            "/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/"
            "recogdrive_agent_cache_dir_train"
        ),
        "nav_train_newvlm.json",
    ),
    (
        Path(
            "/mnt/datasets/simscale/20260709/"
            "recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0_quality"
        ),
        "sim_round0_quality_newvlm.json",
    ),
    (
        Path(
            "/mnt/datasets/simscale/20260709/"
            "recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-1_quality"
        ),
        "sim_round1_quality_newvlm.json",
    ),
]


def builder_names(cache_root: Path) -> list[str]:
    for log_dir in cache_root.iterdir():
        if not log_dir.is_dir():
            continue
        for token_dir in log_dir.iterdir():
            if not token_dir.is_dir():
                continue
            names = sorted(p.name[:-3] for p in token_dir.glob("*.gz"))
            if names:
                return names
    raise RuntimeError(f"no token dirs under {cache_root}")


def scan_log(args):
    log_dir, needed = args
    out = []
    for token_dir in log_dir.iterdir():
        if not token_dir.is_dir():
            continue
        present = {p.name[:-3] for p in token_dir.glob("*.gz")}
        if needed.issubset(present):
            out.append((token_dir.name, f"{log_dir.name}/{token_dir.name}"))
    return out


def build_one(cache_root: Path, out_name: str, workers: int) -> Path:
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache root missing: {cache_root}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[start] {out_name} <- {cache_root}", flush=True)
    needed = set(builder_names(cache_root))
    logs = [p for p in cache_root.iterdir() if p.is_dir()]
    print(f"[info] {out_name}: logs={len(logs)} builders={sorted(needed)} workers={workers}", flush=True)
    tokens = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(scan_log, [(lg, needed) for lg in logs]):
            for token, rel in chunk:
                tokens[token] = rel
            done += 1
            if done % 50 == 0 or done == len(logs):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(logs) - done) / max(rate, 1e-6)
                print(
                    f"[prog] {out_name}: {done}/{len(logs)} logs, "
                    f"tokens={len(tokens)}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                    flush=True,
                )
    manifest = {
        "cache_path": str(cache_root),
        "builders": sorted(needed),
        "tokens": tokens,
    }
    out_path = OUT_DIR / out_name
    out_path.write_text(json.dumps(manifest))
    print(
        f"[done] {out_name}: {len(tokens)} tokens, builders={sorted(needed)}, "
        f"{time.time()-t0:.0f}s -> {out_path}",
        flush=True,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["train", "round0", "round1", "all"],
        default="all",
        help="Which manifest(s) to build.",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    selected = {
        "train": [CACHES[0]],
        "round0": [CACHES[1]],
        "round1": [CACHES[2]],
        "all": CACHES,
    }[args.only]

    for root, out_name in selected:
        build_one(root, out_name, workers=args.workers)


if __name__ == "__main__":
    main()

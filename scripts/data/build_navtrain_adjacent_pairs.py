"""Build the navtrain adjacent-frame pair table for the EC (two-frame extended
comfort) reward term.

A pair (prev_token, cur_token) qualifies when both frames are in the same log,
their timestamp gap is inside (0, 0.55) s -- the official EC validity window --
and both tokens have cached agent features (so they can actually be trained on).

Usage:
  python build_navtrain_adjacent_pairs.py \
      --agent-cache /workspace/datasets/simscale/20260709/new_vlm_vit_hidden_state_nav_sim/recogdrive_agent_cache_dir_train \
      --out /workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1/data/epdms/navtrain_adjacent_pairs.json
"""

import argparse
import json
import pickle
import time
from pathlib import Path

LOGS_DIR = "/workspace/datasets/recdrive/20260513/nby/recdrive/download/navsim_logs/trainval"
MAX_DT_S = 0.55


def normalize_token(token: str) -> str:
    return str(token).strip().lower().replace("-", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=LOGS_DIR)
    parser.add_argument(
        "--agent-cache",
        default="/workspace/datasets/simscale/20260709/new_vlm_vit_hidden_state_nav_sim/recogdrive_agent_cache_dir_train",
    )
    parser.add_argument("--out", default="/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1/data/epdms/navtrain_adjacent_pairs.json")
    args = parser.parse_args()

    t0 = time.time()
    cache_root = Path(args.agent_cache)
    cached = {}  # token -> log_name
    for log_dir in cache_root.iterdir():
        if not log_dir.is_dir():
            continue
        for token_dir in log_dir.iterdir():
            if token_dir.is_dir():
                cached[normalize_token(token_dir.name)] = log_dir.name
    print(f"agent cache: {len(cached)} tokens under {cache_root} ({time.time()-t0:.0f}s)")

    pairs = []
    n_logs = 0
    for pkl_path in sorted(Path(args.logs_dir).glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        n_logs += 1
        rows = sorted(
            (
                (normalize_token(fr["token"]), int(fr["timestamp"]))
                for fr in frames
                if normalize_token(fr["token"]) in cached
            ),
            key=lambda kv: kv[1],
        )
        for (prev_tok, prev_us), (cur_tok, cur_us) in zip(rows[:-1], rows[1:]):
            dt_s = (cur_us - prev_us) / 1e6
            if 0.0 < dt_s < MAX_DT_S:
                pairs.append((prev_tok, cur_tok, round(dt_s, 3)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paired_tokens = {t for p in pairs for t in p[:2]}
    meta = {
        "logs_dir": args.logs_dir,
        "agent_cache": str(cache_root),
        "num_logs": n_logs,
        "num_cached_tokens": len(cached),
        "num_pairs": len(pairs),
        "num_tokens_in_pairs": len(paired_tokens),
        "pair_coverage_of_cache": round(len(paired_tokens) / max(1, len(cached)), 4),
        "max_dt_s": MAX_DT_S,
    }
    out_path.write_text(json.dumps({"meta": meta, "pairs": pairs}))
    print(json.dumps(meta, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

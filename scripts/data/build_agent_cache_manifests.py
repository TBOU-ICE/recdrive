"""Build token manifests for the vit agent-feature caches (one-time, offline).

Walking ~100k token dirs on CPFS at every training launch costs 30-40 min per
rank; with a manifest the dataset resolves tokens in seconds. A token is listed
when its dir contains all builder files (recogdrive_feature.gz / trajectory
target share one file, matching CacheOnlyDataset's validity rule).

Run:  PYTHONPATH=/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1 python build_agent_cache_manifests.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIT_ROOT = Path("/workspace/datasets/simscale/20260709/new_vlm_vit_hidden_state_nav_sim")
OUT_DIR = Path("/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1/data/epdms/manifests")

# builder unique names for ReCogDriveFeatureBuilder + TrajectoryTargetBuilder
BUILDERS = ["recogdrive_feature", "trajectory_target"]

CACHES = [
    ("recogdrive_agent_cache_dir_train", "nav_train_vit.json"),
    ("recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0", "sim_round0_vit.json"),
    ("recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-1", "sim_round1_vit.json"),
]


def builder_names(cache_root: Path) -> list:
    """Infer the actual builder file names from one token dir (robust to naming)."""
    for log_dir in cache_root.iterdir():
        if not log_dir.is_dir():
            continue
        for token_dir in log_dir.iterdir():
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cache_name, out_name in CACHES:
        root = VIT_ROOT / cache_name
        t0 = time.time()
        needed = set(builder_names(root))
        logs = [p for p in root.iterdir() if p.is_dir()]
        tokens = {}
        with ThreadPoolExecutor(max_workers=32) as pool:
            for chunk in pool.map(scan_log, [(lg, needed) for lg in logs]):
                for token, rel in chunk:
                    tokens[token] = rel
        manifest = {"cache_path": str(root), "builders": sorted(needed), "tokens": tokens}
        out_path = OUT_DIR / out_name
        out_path.write_text(json.dumps(manifest))
        print(f"{out_name}: {len(tokens)} tokens, builders={sorted(needed)}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build no-goal bucket IL symlink views and reusable dataset indexes.

Run once on this machine (CPU is enough). Training then loads the JSON indexes
and does not walk Alluxio/cache trees.

  python scripts/data/prep_bucket_il_newvlm.py --all-buckets
  python scripts/data/prep_bucket_il_newvlm.py --bucket-file exclusive_rule_intersection_tokens.json
  python scripts/data/prep_bucket_il_newvlm.py --nav-index-only
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_BUCKETS = [
    "exclusive_safety_dynamics_interaction_tokens.json",
    "exclusive_rule_intersection_tokens.json",
    "exclusive_progress_curbside_stopgo_tokens.json",
    "exclusive_general_or_no_tag_tokens.json",
]
BUCKET_NAME_FROM_FILE = {
    "exclusive_safety_dynamics_interaction_tokens.json": "safety_dynamics_interaction",
    "exclusive_rule_intersection_tokens.json": "rule_intersection",
    "exclusive_progress_curbside_stopgo_tokens.json": "progress_curbside_stopgo",
    "exclusive_general_or_no_tag_tokens.json": "general_or_no_tag",
}
REQUIRED_GZ = ("internvl_feature", "trajectory_target")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_token(token) -> str:
    token = str(token).strip().lower()
    base, sep, suffix = token.rpartition("-")
    if sep and suffix.isdigit() and len(suffix) == 3 and base:
        return token
    return token.replace("-", "")


def load_token_to_log(path: Path) -> dict:
    data = load_json(path)
    out = {}
    for token, meta in data.items():
        if not isinstance(meta, dict):
            continue
        log_name = meta.get("log_name")
        norm = normalize_token(token)
        if norm and log_name:
            out[norm] = log_name
    return out


def token_has_required_gz(token_dir: Path) -> bool:
    return all((token_dir / f"{name}.gz").is_file() for name in REQUIRED_GZ)


def link_one(src_root: Path, dst_root: Path, log_name: str, token: str, trust_src: bool = False) -> bool:
    dst_log = dst_root / log_name
    dst = dst_log / token
    if dst.exists() or dst.is_symlink():
        return True
    src = src_root / log_name / token
    if not trust_src and (not src.is_dir() or not token_has_required_gz(src)):
        return False
    dst_log.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=True)
    except FileExistsError:
        return True
    except (FileNotFoundError, OSError):
        return False
    return True


def link_tokens(
    src_root: Path,
    dst_root: Path,
    token_to_log: dict,
    tokens: list,
    workers: int,
    label: str = "link",
    trust_src: bool = False,
) -> tuple[int, list]:
    linked = 0
    kept = []
    total = len(tokens)
    print(f"[{label}] linking {total} tokens -> {dst_root} (workers={workers})", flush=True)

    def _job(token):
        log_name = token_to_log.get(token)
        if not log_name:
            return None
        if link_one(src_root, dst_root, log_name, token, trust_src=trust_src):
            return token, log_name
        return None

    done = 0
    if workers <= 1:
        iterator = ((_job(t)) for t in tokens)
        for item in iterator:
            done += 1
            if item is not None:
                linked += 1
                kept.append(item)
            if done % 2000 == 0 or done == total:
                print(f"[{label}] {done}/{total} tried, linked={linked}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for item in pool.map(_job, tokens, chunksize=64):
                done += 1
                if item is not None:
                    linked += 1
                    kept.append(item)
                if done % 2000 == 0 or done == total:
                    print(f"[{label}] {done}/{total} tried, linked={linked}", flush=True)
    return linked, kept


def resolve_sim_cache(sim_agent_root: Path, quality_root: Path, dataset_name: str) -> Path:
    quality = sim_agent_root / f"recogdrive_agent_cache_dir_{dataset_name}_quality"
    quality_cpfs = quality_root / f"recogdrive_agent_cache_dir_{dataset_name}_quality"
    full = sim_agent_root / f"recogdrive_agent_cache_dir_{dataset_name}"
    if quality.is_dir():
        return quality
    if quality_cpfs.is_dir():
        return quality_cpfs
    return full


def build_nav_index(nav_cache: Path, out_path: Path, workers: int) -> Path:
    if not nav_cache.is_dir():
        raise RuntimeError(f"NAV cache missing: {nav_cache}")
    logs = [p for p in nav_cache.iterdir() if p.is_dir()]
    print(f"[nav-index] scanning {len(logs)} logs under {nav_cache}", flush=True)
    tokens = {}

    def scan_log(log_dir: Path):
        out = []
        for token_dir in log_dir.iterdir():
            if token_dir.is_dir() and token_has_required_gz(token_dir):
                out.append((token_dir.name, f"{log_dir.name}/{token_dir.name}"))
        return out

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for chunk in pool.map(scan_log, logs):
            for token, rel in chunk:
                tokens[token] = rel
            done += 1
            if done % 50 == 0 or done == len(logs):
                print(f"[nav-index] {done}/{len(logs)} logs, tokens={len(tokens)}", flush=True)

    payload = {
        "cache_path": str(nav_cache),
        "builders": list(REQUIRED_GZ),
        "tokens": tokens,
    }
    save_json(payload, out_path)
    print(f"[nav-index] wrote {len(tokens)} tokens -> {out_path}", flush=True)
    return out_path


def prep_one_bucket(args, bucket_file: str) -> dict:
    bucket_name = BUCKET_NAME_FROM_FILE.get(bucket_file, Path(bucket_file).stem)
    mix_root = Path(args.mix_root_parent) / f"{bucket_name}_fullmix"
    nav_bucket_cache = mix_root / "navtrain_bucket_cache"
    sim_bucket_cache = mix_root / "simscale_bucket_cache"
    info_dir = mix_root / "metadata"
    nav_bucket_cache.mkdir(parents=True, exist_ok=True)
    sim_bucket_cache.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    navtrain_output_dir = Path(args.navtrain_output_dir)
    nav_cache = Path(args.nav_cache)
    sim_rounds = [r.strip() for r in args.sim_rounds.split(",") if r.strip()]
    sim_agent_root = Path(args.sim_agent_cache_root)
    quality_root = Path(args.sim_quality_cache_root)
    bucket_root = Path(args.simscale_bucket_root)

    sim_caches = []
    sim_bucket_dirs = []
    for round_id in sim_rounds:
        dataset_name = f"synthetic_reaction_pdm_v1.0-{round_id}"
        sim_caches.append(resolve_sim_cache(sim_agent_root, quality_root, dataset_name))
        sim_bucket_dirs.append(bucket_root / f"scene_buckets_{dataset_name}_quality")

    nav_token_to_log = load_token_to_log(navtrain_output_dir / "navtrain_token_to_buckets.json")
    nav_tokens = [normalize_token(t) for t in load_json(navtrain_output_dir / bucket_file)]
    nav_tokens = [t for t in nav_tokens if t in nav_token_to_log]
    nav_listed = len(nav_tokens)
    nav_index_path = Path(args.nav_index_path)
    trust_nav = False
    if nav_index_path.is_file():
        nav_index_tokens = set(load_json(nav_index_path).get("tokens", {}))
        before = len(nav_tokens)
        nav_tokens = [t for t in nav_tokens if t in nav_index_tokens]
        trust_nav = True
        print(
            f"[prep] {bucket_file}: nav listed={nav_listed} in_cache_index={len(nav_tokens)} "
            f"(dropped {before - len(nav_tokens)} missing from new-VLM cache)",
            flush=True,
        )
    linked_nav, nav_kept = link_tokens(
        nav_cache,
        nav_bucket_cache,
        nav_token_to_log,
        nav_tokens,
        args.workers,
        label=f"{bucket_name}-nav",
        trust_src=trust_nav,
    )
    if linked_nav == 0:
        raise RuntimeError(f"No navtrain links for {bucket_file} from {nav_cache}")

    sim_kept = []
    linked_sim = 0
    linked_sim_by_round = {}
    for sim_bucket_dir, sim_cache in zip(sim_bucket_dirs, sim_caches):
        token_map_path = sim_bucket_dir / "simscale_token_to_buckets.json"
        rule_path = sim_bucket_dir / bucket_file
        if not token_map_path.is_file():
            raise RuntimeError(f"Missing SimScale token map: {token_map_path}")
        if not rule_path.is_file():
            raise RuntimeError(f"Missing SimScale bucket file: {rule_path}")
        if not sim_cache.is_dir():
            raise RuntimeError(f"Missing SimScale agent cache: {sim_cache}")
        sim_token_to_log = load_token_to_log(token_map_path)
        round_tokens = [normalize_token(t) for t in load_json(rule_path)]
        round_tokens = [t for t in round_tokens if t in sim_token_to_log]
        linked_round, kept_round = link_tokens(
            sim_cache,
            sim_bucket_cache,
            sim_token_to_log,
            round_tokens,
            args.workers,
            label=f"{bucket_name}-sim-{sim_bucket_dir.name}",
        )
        linked_sim += linked_round
        sim_kept.extend(kept_round)
        linked_sim_by_round[str(sim_bucket_dir)] = {
            "bucket_tokens": len(round_tokens),
            "linked": linked_round,
            "agent_cache": str(sim_cache),
        }
    if linked_sim == 0:
        raise RuntimeError("No SimScale bucket cache entries were linked.")

    train_samples = []
    for token, log_name in nav_kept:
        train_samples.append(
            {
                "source": "navtrain_bucket",
                "token": token,
                "path": str(nav_bucket_cache / log_name / token),
            }
        )
    for token, log_name in sim_kept:
        train_samples.append(
            {
                "source": "simscale_bucket",
                "token": token,
                "path": str(sim_bucket_cache / log_name / token),
            }
        )

    train_index = {
        "builders": list(REQUIRED_GZ),
        "sources": {
            "navtrain_bucket": {"cache_path": str(nav_bucket_cache), "count": linked_nav},
            "simscale_bucket": {"cache_path": str(sim_bucket_cache), "count": linked_sim},
        },
        "samples": train_samples,
    }
    train_index_path = info_dir / "train_index.json"
    save_json(train_index, train_index_path)

    summary = {
        "stage": "il",
        "bucket_file": bucket_file,
        "bucket_name": bucket_name,
        "nav_bucket_tokens": nav_listed,
        "nav_bucket_tokens_in_cache": linked_nav,
        "sim_bucket_tokens": len(sim_kept),
        "linked_nav_bucket": linked_nav,
        "linked_sim_bucket": linked_sim,
        "linked_sim_by_round": linked_sim_by_round,
        "nav_bucket_cache": str(nav_bucket_cache),
        "sim_bucket_cache": str(sim_bucket_cache),
        "train_index": str(train_index_path),
        "train_index_samples": len(train_samples),
    }
    save_json(summary, info_dir / "bucket_il_sources_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all-buckets", action="store_true")
    p.add_argument("--bucket-file", default=os.environ.get("MIX_BUCKET_FILE", ""))
    p.add_argument("--nav-index-only", action="store_true")
    p.add_argument(
        "--navtrain-output-dir",
        default=os.environ.get(
            "MIX_NAVTRAIN_OUTPUT_DIR",
            "/mnt/datasets/simscale/20260709/data/navtrain_scene/output/navtrain",
        ),
    )
    p.add_argument(
        "--nav-cache",
        default=os.environ.get(
            "MIX_NAV_CACHE_PATH",
            "/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train",
        ),
    )
    p.add_argument(
        "--sim-agent-cache-root",
        default="/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim",
    )
    p.add_argument(
        "--sim-quality-cache-root",
        default="/mnt/datasets/simscale/20260709/data/simscale/new_vlm_quality_views",
    )
    p.add_argument(
        "--simscale-bucket-root",
        default="/mnt/datasets/simscale/20260709/data/simscale",
    )
    p.add_argument(
        "--mix-root-parent",
        default="/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm",
    )
    p.add_argument("--sim-rounds", default=os.environ.get("SIM_ROUNDS", "0,1"))
    p.add_argument(
        "--nav-index-path",
        default="/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm/navtrain_cache_index.json",
    )
    p.add_argument("--workers", type=int, default=int(os.environ.get("PREP_WORKERS", "16")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nav_index_path = Path(args.nav_index_path)
    if args.nav_index_only or args.all_buckets:
        if not nav_index_path.is_file():
            build_nav_index(Path(args.nav_cache), nav_index_path, args.workers)
        else:
            print(f"[nav-index] already exists: {nav_index_path}", flush=True)
        if args.nav_index_only:
            return

    if args.all_buckets:
        bucket_files = DEFAULT_BUCKETS
    elif args.bucket_file:
        bucket_files = [args.bucket_file]
    else:
        raise SystemExit("Pass --all-buckets or --bucket-file")

    for bucket_file in bucket_files:
        print(f"===== prep {bucket_file} =====", flush=True)
        prep_one_bucket(args, bucket_file)


if __name__ == "__main__":
    main()

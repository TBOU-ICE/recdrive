#!/usr/bin/env python3
"""Build portable bucket indexes with direct /mnt cache paths and no links."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
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
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_token(token) -> str:
    token = str(token).strip().lower()
    base, separator, suffix = token.rpartition("-")
    if separator and suffix.isdigit() and len(suffix) == 3 and base:
        return token
    return token.replace("-", "")


def load_token_to_log(path: Path) -> dict[str, str]:
    output = {}
    for token, metadata in load_json(path).items():
        if not isinstance(metadata, dict):
            continue
        normalized = normalize_token(token)
        log_name = metadata.get("log_name")
        if normalized and log_name:
            output[normalized] = str(log_name)
    return output


def token_has_required_gz(token_dir: Path) -> bool:
    return token_dir.is_dir() and all(
        (token_dir / f"{builder}.gz").is_file() for builder in REQUIRED_GZ
    )


def scan_nav_cache(nav_cache: Path, workers: int) -> dict[str, str]:
    logs = [path for path in nav_cache.iterdir() if path.is_dir()]
    print(f"[nav-index] scanning {len(logs)} logs under {nav_cache}", flush=True)

    def scan_log(log_dir: Path) -> list[tuple[str, str]]:
        return [
            (token_dir.name, f"{log_dir.name}/{token_dir.name}")
            for token_dir in log_dir.iterdir()
            if token_has_required_gz(token_dir)
        ]

    tokens = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, chunk in enumerate(pool.map(scan_log, logs), start=1):
            tokens.update(chunk)
            if done % 50 == 0 or done == len(logs):
                print(f"[nav-index] {done}/{len(logs)} logs, tokens={len(tokens)}", flush=True)
    return tokens


def load_or_build_nav_tokens(args) -> dict[str, str]:
    source_index = Path(args.nav_source_index)
    if source_index.is_file():
        tokens = {
            normalize_token(token): str(relative_path)
            for token, relative_path in load_json(source_index).get("tokens", {}).items()
        }
        print(f"[nav-index] reused {len(tokens)} entries from {source_index}", flush=True)
        return tokens
    return scan_nav_cache(Path(args.nav_cache), args.workers)


def validate_samples(
    cache_root: Path,
    token_to_log: dict[str, str],
    tokens: list[str],
    workers: int,
    label: str,
) -> list[tuple[str, str]]:
    candidates = [
        (token, token_to_log[token])
        for token in tokens
        if token in token_to_log
    ]

    def validate(item: tuple[str, str]):
        token, log_name = item
        return item if token_has_required_gz(cache_root / log_name / token) else None

    kept = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, item in enumerate(pool.map(validate, candidates, chunksize=64), start=1):
            if item is not None:
                kept.append(item)
            if done % 2000 == 0 or done == len(candidates):
                print(f"[{label}] {done}/{len(candidates)} checked, valid={len(kept)}", flush=True)
    return kept


def prepare_bucket(
    args,
    bucket_file: str,
    nav_tokens_by_path: dict[str, str],
    sim_caches: list[Path],
) -> dict:
    bucket_name = BUCKET_NAME_FROM_FILE.get(bucket_file, Path(bucket_file).stem)
    output_dir = Path(args.output_root) / bucket_name
    nav_output_dir = Path(args.navtrain_output_dir)
    sim_bucket_root = Path(args.simscale_bucket_root)

    requested_nav_tokens = [
        normalize_token(token) for token in load_json(nav_output_dir / bucket_file)
    ]
    valid_nav_tokens = [
        token for token in requested_nav_tokens if token in nav_tokens_by_path
    ]

    train_samples = [
        {
            "source": "navtrain_bucket",
            "token": token,
            "path": str(Path(args.nav_cache) / nav_tokens_by_path[token]),
        }
        for token in valid_nav_tokens
    ]
    source_counts = {"navtrain_bucket": len(train_samples), "simscale_bucket": 0}
    sim_round_counts = {}

    for round_id, cache_root in zip(args.sim_rounds, sim_caches):
        bucket_dir = (
            sim_bucket_root
            / f"scene_buckets_synthetic_reaction_pdm_v1.0-{round_id}_quality"
        )
        token_map = load_token_to_log(bucket_dir / "simscale_token_to_buckets.json")
        requested_tokens = [
            normalize_token(token) for token in load_json(bucket_dir / bucket_file)
        ]
        valid_samples = validate_samples(
            cache_root,
            token_map,
            requested_tokens,
            args.workers,
            f"{bucket_name}-sim-{round_id}",
        )
        train_samples.extend(
            {
                "source": "simscale_bucket",
                "token": token,
                "path": str(cache_root / log_name / token),
            }
            for token, log_name in valid_samples
        )
        sim_round_counts[str(round_id)] = len(valid_samples)
        source_counts["simscale_bucket"] += len(valid_samples)

    if not train_samples or source_counts["navtrain_bucket"] == 0:
        raise RuntimeError(f"No valid training samples for bucket {bucket_name}")

    train_index = {
        "format": "direct-absolute-paths-v1",
        "builders": list(REQUIRED_GZ),
        "sources": {
            "navtrain_bucket": {
                "cache_path": str(Path(args.nav_cache)),
                "count": source_counts["navtrain_bucket"],
            },
            "simscale_bucket": {
                "cache_paths": [str(path) for path in sim_caches],
                "count": source_counts["simscale_bucket"],
            },
        },
        "samples": train_samples,
    }
    train_index_path = output_dir / "train_index.json"
    save_json(train_index, train_index_path)

    bucket_val_tokens = {
        token: nav_tokens_by_path[token]
        for token in valid_nav_tokens
    }
    bucket_val_index_path = output_dir / "navtrain_bucket_val_index.json"
    save_json(
        {
            "format": "direct-absolute-root-v1",
            "cache_path": str(Path(args.nav_cache)),
            "builders": list(REQUIRED_GZ),
            "tokens": bucket_val_tokens,
        },
        bucket_val_index_path,
    )

    summary = {
        "stage": "il",
        "storage": "direct_absolute_paths",
        "bucket_file": bucket_file,
        "bucket_name": bucket_name,
        "nav_bucket_tokens": len(requested_nav_tokens),
        "nav_bucket_tokens_in_cache": len(valid_nav_tokens),
        "sim_bucket_tokens_in_cache": source_counts["simscale_bucket"],
        "sim_round_counts": sim_round_counts,
        "train_index": str(train_index_path),
        "bucket_val_index": str(bucket_val_index_path),
        "train_index_samples": len(train_samples),
    }
    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-buckets", action="store_true")
    parser.add_argument("--bucket-file", default="")
    parser.add_argument(
        "--navtrain-output-dir",
        default="/mnt/datasets/simscale/20260709/data/navtrain_scene/output/navtrain",
    )
    parser.add_argument(
        "--nav-cache",
        default="/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train",
    )
    parser.add_argument(
        "--nav-source-index",
        default="/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm/navtrain_cache_index.json",
    )
    parser.add_argument(
        "--sim-cache",
        action="append",
        default=[],
        help="One original SimScale cache root per --sim-rounds entry.",
    )
    parser.add_argument(
        "--sim-agent-cache-root",
        default="/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim",
    )
    parser.add_argument(
        "--simscale-bucket-root",
        default="/mnt/datasets/simscale/20260709/data/simscale",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm_direct",
    )
    parser.add_argument("--sim-rounds", default=os.environ.get("SIM_ROUNDS", "0,1"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PREP_WORKERS", "16")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.sim_rounds = [
        round_id.strip() for round_id in args.sim_rounds.split(",") if round_id.strip()
    ]
    sim_caches = [Path(path) for path in args.sim_cache]
    if not sim_caches:
        sim_caches = [
            Path(args.sim_agent_cache_root)
            / f"recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-{round_id}"
            for round_id in args.sim_rounds
        ]
    if len(sim_caches) != len(args.sim_rounds):
        raise SystemExit("--sim-cache count must match --sim-rounds count")
    for cache_root in [Path(args.nav_cache), *sim_caches]:
        if not cache_root.is_dir():
            raise SystemExit(f"Cache root does not exist: {cache_root}")

    nav_tokens_by_path = load_or_build_nav_tokens(args)
    full_val_path = Path(args.output_root) / "navtrain_full_val_index.json"
    save_json(
        {
            "format": "direct-absolute-root-v1",
            "cache_path": str(Path(args.nav_cache)),
            "builders": list(REQUIRED_GZ),
            "tokens": nav_tokens_by_path,
        },
        full_val_path,
    )
    print(f"[nav-index] wrote full validation index -> {full_val_path}", flush=True)

    if args.all_buckets:
        bucket_files = DEFAULT_BUCKETS
    elif args.bucket_file:
        bucket_files = [args.bucket_file]
    else:
        raise SystemExit("Pass --all-buckets or --bucket-file")

    for bucket_file in bucket_files:
        print(f"===== prepare direct index: {bucket_file} =====", flush=True)
        prepare_bucket(args, bucket_file, nav_tokens_by_path, sim_caches)


if __name__ == "__main__":
    main()

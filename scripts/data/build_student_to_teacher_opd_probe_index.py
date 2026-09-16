#!/usr/bin/env python3
"""Build the fixed 4-bucket Phase-C probe from held-out navtrain validation.

Selection is stable across machines: samples are ordered by SHA256(seed, bucket,
token), not by filesystem traversal or Python's process-randomised ``hash``.
The output stores absolute cache paths and per-token diffusion seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set

import yaml


BUCKETS = (
    "general_or_no_tag",
    "progress_curbside_stopgo",
    "rule_intersection",
    "safety_dynamics_interaction",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, bucket: str, token: str) -> str:
    return hashlib.sha256(f"{seed}\0{bucket}\0{token}".encode("utf-8")).hexdigest()


def _noise_seed(seed: int, token: str) -> int:
    raw = hashlib.sha256(f"noise\0{seed}\0{token}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % (2**31 - 1)


def _load_train_tokens(
    index_root: Path,
    index_name: str,
    allowed_navtrain_logs: Set[str],
) -> Set[str]:
    tokens: Set[str] = set()
    for bucket in BUCKETS:
        path = index_root / bucket / index_name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "direct-absolute-paths-v1":
            raise ValueError(f"Unexpected index format in {path}: {payload.get('format')!r}")
        for sample in payload.get("samples", []):
            if sample.get("source") == "navtrain_bucket":
                log_name = Path(str(sample["path"])).parent.name
                if log_name not in allowed_navtrain_logs:
                    continue
            tokens.add(str(sample["token"]))
    return tokens


def _absolute_val_path(cache_root: Path, relative_or_absolute: str) -> Path:
    path = Path(str(relative_or_absolute))
    return path if path.is_absolute() else cache_root / path


def _portable_source_label(path: Path) -> str:
    """Prefer a repo-relative config label so rebuilt JSON is machine-stable."""
    parts = path.resolve().parts
    if "navsim" in parts:
        return Path(*parts[parts.index("navsim") :]).as_posix()
    return str(path)


def build_probe(
    *,
    val_index: Path,
    bucket_map_path: Path,
    split_config: Path,
    index_root: Path,
    index_name: str,
    per_bucket: int,
    viz_per_bucket: int,
    seed: int,
) -> Dict[str, object]:
    val_payload = json.loads(val_index.read_text(encoding="utf-8"))
    val_tokens = val_payload.get("tokens", {})
    cache_root_value = str(val_payload.get("cache_path", "")).strip()
    cache_root = Path(cache_root_value)
    if not val_tokens or not cache_root_value:
        raise ValueError(f"Validation index is missing cache_path/tokens: {val_index}")

    bucket_map = json.loads(bucket_map_path.read_text(encoding="utf-8"))
    split_payload = yaml.safe_load(split_config.read_text(encoding="utf-8"))
    train_log_names = {str(name) for name in split_payload.get("train_logs", [])}
    val_log_names = {str(name) for name in split_payload.get("val_logs", [])}
    if not train_log_names or not val_log_names:
        raise ValueError(f"Missing train_logs/val_logs in {split_config}")
    if train_log_names & val_log_names:
        raise ValueError(
            f"Configured train/val logs overlap by {len(train_log_names & val_log_names)}"
        )
    train_tokens = _load_train_tokens(index_root, index_name, train_log_names)
    candidates: Dict[str, List[str]] = {bucket: [] for bucket in BUCKETS}
    overlap = 0
    n_official_val = 0
    for token, relative in val_tokens.items():
        token = str(token)
        log_name = str(relative).split("/", 1)[0]
        if log_name not in val_log_names:
            continue
        n_official_val += 1
        if token in train_tokens:
            overlap += 1
            continue
        bucket = bucket_map.get(token)
        if bucket in candidates:
            candidates[bucket].append(token)

    if overlap:
        raise RuntimeError(
            f"Validation index overlaps the four training indexes by {overlap} token(s); "
            "refusing to build a leaky probe."
        )

    samples: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}
    for bucket in BUCKETS:
        ordered = sorted(candidates[bucket], key=lambda token: _stable_key(seed, bucket, token))
        if len(ordered) < per_bucket:
            raise RuntimeError(
                f"Bucket {bucket!r} has only {len(ordered)} held-out candidates, "
                f"need {per_bucket}."
            )
        selected = ordered[:per_bucket]
        viz_tokens = set(selected[:viz_per_bucket])
        for token in selected:
            token_path = _absolute_val_path(cache_root, str(val_tokens[token]))
            samples.append(
                {
                    "source": "navtrain_val",
                    "bucket": bucket,
                    "token": token,
                    "path": str(token_path),
                    "noise_seed": _noise_seed(seed, token),
                    "draw_viz": token in viz_tokens,
                }
            )
        counts[bucket] = len(selected)

    return {
        "format": "student-to-teacher-opd-probe-v1",
        "seed": seed,
        "per_bucket": per_bucket,
        "viz_per_bucket": viz_per_bucket,
        "buckets": list(BUCKETS),
        "counts": counts,
        "source_val_index": str(val_index),
        "source_val_index_sha256": _sha256_file(val_index),
        "source_bucket_map": str(bucket_map_path),
        "source_bucket_map_sha256": _sha256_file(bucket_map_path),
        "source_split_config": _portable_source_label(split_config),
        "source_split_config_sha256": _sha256_file(split_config),
        "official_val_candidates": n_official_val,
        "samples": samples,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path(
            "/mnt/datasets/simscale/20260709/data/simscale/"
            "il_training_newvlm_direct"
        ),
    )
    parser.add_argument("--index-name", default="train_index.json")
    parser.add_argument("--val-index", type=Path, default=None)
    parser.add_argument(
        "--bucket-map",
        type=Path,
        default=Path(
            "/mnt/datasets/simscale/20260709/data/navtrain_scene/output/"
            "navtrain/exclusive_token_to_bucket.json"
        ),
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=repo_root
        / "navsim"
        / "planning"
        / "script"
        / "config"
        / "training"
        / "default_train_val_test_log_split.yaml",
        help="Official train_logs/val_logs boundary; the full index alone is not a split.",
    )
    parser.add_argument("--per-bucket", type=int, default=32)
    parser.add_argument("--viz-per-bucket", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "data"
        / "phase_c"
        / "student_to_teacher_opd_probe_4x32_seed0.json",
    )
    args = parser.parse_args()
    val_index = args.val_index or args.index_root / "navtrain_full_val_index.json"
    if args.per_bucket <= 0:
        raise ValueError("--per-bucket must be positive")
    if not 0 <= args.viz_per_bucket <= args.per_bucket:
        raise ValueError("--viz-per-bucket must be in [0, per-bucket]")

    payload = build_probe(
        val_index=val_index,
        bucket_map_path=args.bucket_map,
        split_config=args.split_config,
        index_root=args.index_root,
        index_name=args.index_name,
        per_bucket=args.per_bucket,
        viz_per_bucket=args.viz_per_bucket,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    digest = _sha256_file(args.output)
    print(f"wrote {args.output}")
    print(f"samples={len(payload['samples'])} counts={payload['counts']}")
    print(f"probe_sha256={digest}")


if __name__ == "__main__":
    main()

"""Create log-disjoint direct indexes for student-state correction.

The goal-OPD bucket ``train_index.json`` files contain both NAVSIM train and
validation logs (plus SimScale training samples).  The state-correction runner
intentionally rejects that leakage.  This utility applies NAVSIM's canonical
train/validation log split and writes two direct indexes that satisfy the
runner's stricter contract.

Example:
    python scripts/data/prepare_state_correction_indexes.py \
      --bucket general_or_no_tag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECT_ROOT = Path(
    "/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm_direct"
)
DEFAULT_SPLIT = (
    REPO_ROOT
    / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/state_correction/indexes"
def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    payload = json.loads(path.read_text())
    builders = payload.get("builders", [])
    if "samples" in payload:
        rows = payload["samples"]
    else:
        cache_root = Path(payload["cache_path"])
        rows = [
            {"token": str(token), "path": str(cache_root / relative)}
            for token, relative in payload.get("tokens", {}).items()
        ]
    return builders, rows


def log_name(row: dict[str, str]) -> str:
    return Path(row["path"]).parent.name


def write_index(path: Path, builders: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "recdrive_direct_cache_index_v1",
        "builders": builders,
        "samples": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def read_bad_cache_dirs(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"Known-bad cache list is missing: {path}")
    bad_dirs: set[str] = set()
    for line in path.read_text().splitlines():
        cache_file = line.split("\t", 1)[0].strip()
        if not cache_file:
            continue
        directory = str(Path(cache_file).parent)
        bad_dirs.add(directory)
        if directory.startswith("/workspace/datasets/"):
            bad_dirs.add(directory.replace("/workspace/datasets/", "/mnt/datasets/", 1))
        elif directory.startswith("/mnt/datasets/"):
            bad_dirs.add(directory.replace("/mnt/datasets/", "/workspace/datasets/", 1))
    return bad_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="general_or_no_tag")
    parser.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bad-cache-list",
        type=Path,
        default=None,
        help="Optional known corrupt/hanging cache-shard list to omit.",
    )
    args = parser.parse_args()

    bucket_root = args.direct_root / args.bucket
    train_builders, train_rows = read_rows(bucket_root / "train_index.json")
    val_builders, val_rows = read_rows(bucket_root / "navtrain_bucket_val_index.json")
    if sorted(train_builders) != sorted(val_builders):
        raise ValueError("Train and validation indexes use different builders")

    split = yaml.safe_load(args.split.read_text())
    train_logs = set(split["train_logs"])
    val_logs = set(split["val_logs"])
    if train_logs & val_logs:
        raise ValueError("Canonical train/validation log lists overlap")
    bad_cache_dirs = read_bad_cache_dirs(args.bad_cache_list)

    # SimScale samples are training-only. NAVSIM samples follow the canonical
    # log split, regardless of whether an older direct index labeled all of
    # them as training samples.
    filtered_train = [
        row
        for row in train_rows
        if (row.get("source") == "simscale_bucket" or log_name(row) in train_logs)
        and row["path"] not in bad_cache_dirs
    ]
    filtered_val = [
        row
        for row in val_rows
        if log_name(row) in val_logs and row["path"] not in bad_cache_dirs
    ]
    if not filtered_train or not filtered_val:
        raise ValueError("Filtering produced an empty split")

    train_tokens = {row["token"] for row in filtered_train}
    val_tokens = {row["token"] for row in filtered_val}
    train_output_logs = {log_name(row) for row in filtered_train}
    val_output_logs = {log_name(row) for row in filtered_val}
    if train_tokens & val_tokens:
        raise ValueError("Filtered train/validation tokens overlap")
    if train_output_logs & val_output_logs:
        raise ValueError("Filtered train/validation logs overlap")

    train_out = args.output_dir / f"{args.bucket}_train_index.json"
    val_out = args.output_dir / f"{args.bucket}_val_index.json"
    write_index(train_out, train_builders, filtered_train)
    write_index(val_out, val_builders, filtered_val)
    print(
        f"wrote {len(filtered_train)} train samples to {train_out}\n"
        f"wrote {len(filtered_val)} validation samples to {val_out}"
    )


if __name__ == "__main__":
    main()

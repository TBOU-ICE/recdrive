#!/usr/bin/env python3
"""
Merge a full InternVL HuggingFace export (base) with a finetuning checkpoint dir
(weights + tokenizer/config only) into one directory suitable for agent.vlm_path.

Base must contain configuration_internvl_chat.py and modeling_internvl_chat.py.
Checkpoint typically has model.safetensors (or sharded) and tokenizer files.

Example:
  python scripts/evaluation/merge_internvl_base_and_checkpoint.py \\
    --base /path/to/InternVL3-2B \\
    --checkpoint /path/to/checkpoint-400 \\
    --output /path/to/InternVL3-2B_ckpt400_merged
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


# Files/dirs from checkpoint we may overlay (training junk excluded).
CHECKPOINT_WEIGHT_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
)
CHECKPOINT_OPTIONAL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


def _copytree_merge(src: Path, dst: Path) -> None:
    """Copy src tree into dst; existing files under dst are overwritten when copied from src."""
    for root, dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (target_dir / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            t = target_dir / f
            shutil.copy2(s, t)


def _shard_files_from_index(checkpoint: Path) -> list[Path]:
    index = checkpoint / "model.safetensors.index.json"
    if not index.is_file():
        return []
    with open(index, encoding="utf-8") as fp:
        weight_map = json.load(fp).get("weight_map", {})
    names = sorted(set(weight_map.values()))
    return [checkpoint / n for n in names]


def _require_base_layout(base: Path) -> None:
    for name in ("configuration_internvl_chat.py", "modeling_internvl_chat.py", "config.json"):
        p = base / name
        if not p.is_file():
            raise SystemExit(f"Base model dir missing required file: {p}")


def _require_checkpoint_weights(checkpoint: Path) -> None:
    single = checkpoint / "model.safetensors"
    index = checkpoint / "model.safetensors.index.json"
    if single.is_file():
        return
    if index.is_file():
        shards = _shard_files_from_index(checkpoint)
        missing = [str(s) for s in shards if not s.is_file()]
        if missing:
            raise SystemExit(
                "Sharded checkpoint index lists files that are missing:\n" + "\n".join(missing)
            )
        return
    raise SystemExit(
        f"No weights found under {checkpoint}: expected model.safetensors "
        "or model.safetensors.index.json + shard files."
    )


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="Full InternVL HF directory")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Finetune checkpoint dir")
    ap.add_argument("--output", type=Path, required=True, help="Merged output directory")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output dir (updates files in place)",
    )
    args = ap.parse_args(list(argv))

    base = args.base.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()

    if not base.is_dir():
        raise SystemExit(f"Base path is not a directory: {base}")
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint path is not a directory: {checkpoint}")

    _require_base_layout(base)
    _require_checkpoint_weights(checkpoint)

    if output.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output}. Use --overwrite to update, "
                "or choose a new --output path."
            )
    else:
        output.mkdir(parents=True)

    print(f"Copying base: {base} -> {output}")
    _copytree_merge(base, output)

    print(f"Overlaying checkpoint artifacts from: {checkpoint}")
    # Weights
    w_single = checkpoint / "model.safetensors"
    if w_single.is_file():
        shutil.copy2(w_single, output / "model.safetensors")
    w_index = checkpoint / "model.safetensors.index.json"
    if w_index.is_file():
        shutil.copy2(w_index, output / "model.safetensors.index.json")
        for shard in _shard_files_from_index(checkpoint):
            if shard.is_file():
                shutil.copy2(shard, output / shard.name)

    # Config / tokenizer (optional but recommended from checkpoint)
    for name in CHECKPOINT_OPTIONAL_FILES:
        src = checkpoint / name
        if src.is_file():
            shutil.copy2(src, output / name)

    print("Done.")
    print(f"Use agent.vlm_path={output}")
    print("agent.vlm_weights_path can be omitted (weights already merged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

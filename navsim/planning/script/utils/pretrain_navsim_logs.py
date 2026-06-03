"""
Derive NuPlan / NavSim log folder names from InternVL pretrain meta (recogdrive_pretrain.json).

Used to align OPD cache-only training with the Navsim + Navsim_QA splits referenced in
``internvl_chat/shell/data_info/recogdrive_pretrain.json``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# e.g. 2021.05.12.19.36.12_veh-35_00005_00204
_LOG_PART = re.compile(
    r"\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+_\d+_\d+"
)


def _log_name_from_path(path: str) -> Optional[str]:
    norm = path.replace("\\", "/")
    m = _LOG_PART.search(norm)
    if m:
        return m.group(0)
    for part in Path(norm).parts:
        if "_veh-" in part and part[0].isdigit():
            return part
    return None


def _paths_from_record(record: dict) -> List[str]:
    img = record.get("image")
    if isinstance(img, str):
        return [img]
    if isinstance(img, list):
        return [str(x) for x in img if x]
    return []


def _collect_logs_from_jsonl(jsonl_path: Path, max_lines: Optional[int] = None) -> Set[str]:
    logs: Set[str] = set()
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for p in _paths_from_record(rec):
                name = _log_name_from_path(p)
                if name:
                    logs.add(name)
            n += 1
            if max_lines is not None and n >= max_lines:
                break
    return logs


def logs_from_pretrain_meta_json(
    meta_json: str | Path,
    dataset_keys: Iterable[str] = ("Navsim", "Navsim_QA"),
    max_lines_per_jsonl: Optional[int] = None,
    val_num_logs: int = 64,
) -> Tuple[List[str], List[str]]:
    """
    :param meta_json: Path to recogdrive_pretrain.json
    :param dataset_keys: Which meta entries to scan (default: Navsim + Navsim_QA)
    :param max_lines_per_jsonl: Optional cap per jsonl for quick tests
    :param val_num_logs: Number of log folders to use for val (tail of sorted list)
    :return: (train_logs, val_logs) sorted lists of log directory names
    """
    meta_path = Path(meta_json)
    if not meta_path.is_file():
        raise FileNotFoundError(f"pretrain meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    all_logs: Set[str] = set()

    for key in dataset_keys:
        block = meta.get(key)
        if not isinstance(block, dict):
            continue
        ann = block.get("annotation")
        if not ann:
            continue
        ap = Path(ann)
        if not ap.is_file():
            logger.warning("Annotation missing (skip %s): %s", key, ap)
            continue
        found = _collect_logs_from_jsonl(ap, max_lines=max_lines_per_jsonl)
        logger.info("From %s (%s): %d unique log names", key, ap.name, len(found))
        all_logs |= found

    if not all_logs:
        raise RuntimeError(
            f"No log names extracted from {meta_path}. Check jsonl paths and image fields."
        )

    sorted_logs = sorted(all_logs)
    k = min(val_num_logs, len(sorted_logs))
    val_logs = sorted_logs[-k:] if k else sorted_logs
    train_logs = sorted_logs
    return train_logs, val_logs

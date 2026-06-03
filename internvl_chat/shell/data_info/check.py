#!/usr/bin/env python3
"""
根据 recogdrive_pretrain.json 中的 annotation 路径，检查 jsonl 是否含错拼 nuscenessamples。
可选：--fix 将 nuscenessamples 替换为 nuscenes/samples（先备份 .bak）。
用法:
  python3 check_fix_nuscenes_typos_in_jsonl.py /path/to/recogdrive_pretrain.json
  python3 check_fix_nuscenes_typos_in_jsonl.py /path/to/recogdrive_pretrain.json --fix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WRONG = "nuscenessamples"
RIGHT = "nuscenes/samples"


def load_annotation_paths(json_path: Path) -> list[str]:
    with json_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    paths = []
    for name, meta in cfg.items():
        if isinstance(meta, dict) and "annotation" in meta:
            paths.append((name, meta["annotation"]))
    return paths


def scan_file(path: Path) -> tuple[int, list[str]]:
    """返回 (命中行数, 前几条样例行文本)."""
    hits: list[str] = []
    count = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return -1, [f"<读取失败: {e}>"]
    for line in text.splitlines():
        if WRONG in line:
            count += 1
            if len(hits) < 3:
                snippet = line.strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                hits.append(snippet)
    return count, hits


def fix_file(path: Path) -> tuple[bool, str]:
    """原地替换；成功则写回并留 .bak。返回 (是否修改, 说明)."""
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        data = path.read_text(encoding="utf-8", errors="strict")
    except OSError as e:
        return False, f"读取失败: {e}"
    if WRONG not in data:
        return False, "无错拼，跳过"
    if not bak.exists():
        bak.write_text(data, encoding="utf-8")
    fixed = data.replace(WRONG, RIGHT)
    path.write_text(fixed, encoding="utf-8")
    return True, f"已替换并备份 -> {bak}"


def main() -> int:
    ap = argparse.ArgumentParser(description="检查/修复 jsonl 中的 nuscenessamples 错拼")
    ap.add_argument(
        "pretrain_json",
        type=Path,
        nargs="?",
        default=Path("/workspace/recogdrive/internvl_chat/shell/data_info/recogdrive_pretrain.json"),
        help="recogdrive_pretrain.json 路径",
    )
    ap.add_argument("--fix", action="store_true", help="执行替换（nuscenessamples -> nuscenes/samples）并备份 .bak")
    args = ap.parse_args()

    if not args.pretrain_json.is_file():
        print(f"找不到文件: {args.pretrain_json}", file=sys.stderr)
        return 1

    items = load_annotation_paths(args.pretrain_json)
    print(f"配置项数: {len(items)}\n")

    any_hit = False
    for dataset_name, ann in sorted(items, key=lambda x: x[0]):
        p = Path(ann)
        print(f"[{dataset_name}]")
        print(f"  annotation: {ann}")
        if not p.is_file():
            print("  状态: 文件不存在（跳过，请在有数据的机器上跑）\n")
            continue
        n, samples = scan_file(p)
        if n < 0:
            print(f"  状态: {samples[0]}\n")
            continue
        if n == 0:
            print("  错拼命中: 0\n")
            continue
        any_hit = True
        print(f"  错拼命中行数: {n}")
        for i, s in enumerate(samples, 1):
            print(f"  样例{i}: {s}")
        if args.fix:
            ok, msg = fix_file(p)
            print(f"  修复: {msg}")
        print()

    if not args.fix and not any_hit:
        print("全部已检查文件中均未发现 nuscenessamples（或文件均不存在）。")
    elif args.fix:
        print("若需从备份恢复: mv file.jsonl.bak file.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
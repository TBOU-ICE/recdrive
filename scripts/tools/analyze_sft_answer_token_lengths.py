#!/usr/bin/env python3
"""
Estimate assistant / answer token lengths in SFT jsonl (same schema as InternVL LazySupervisedDataset).

- **Assistant text** is whatever appears in the last `gpt`/`assistant` turn; if the label contains
  chain-of-thought before the final answer, those tokens are included in `answer_token_count`.
- **max_seq_length** in InternVL (e.g. 12288) is an upper bound on the *full* training sequence
  (system + user + images placeholders + assistant), not `opd_max_new_tokens` in NavSim OPD.

Examples:
  python scripts/tools/analyze_sft_answer_token_lengths.py \\
    --model_path /mnt/volumes/.../InternVL3-2B \\
    --meta_json internvl_chat/shell/data_info/recogdrive_pretrain.json \\
    --datasets Navsim_QA Navsim \\
    --max_samples 2000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer


def extract_prompt_answer(record: Dict[str, Any]) -> Tuple[str, str]:
    conv = record.get("conversations") or record.get("messages")
    prompt, answer = "", ""
    if isinstance(conv, list):
        for item in conv:
            if not isinstance(item, dict):
                continue
            role = str(item.get("from", item.get("role", ""))).lower()
            text = item.get("value", item.get("content", item.get("text", "")))
            if not isinstance(text, str):
                continue
            if role in {"human", "user"}:
                prompt = text
            elif role in {"gpt", "assistant"}:
                answer = text
    return prompt, answer


def iter_jsonl(path: Path, max_lines: Optional[int]) -> Iterable[str]:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line
            n += 1
            if max_lines is not None and n >= max_lines:
                break


def percentile(xs: List[int], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    c = min(c, len(s) - 1)
    if f == c:
        return float(s[int(round(k))])
    return float(s[f] + (k - f) * (s[c] - s[f]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="HF model dir for tokenizer")
    ap.add_argument("--jsonl_path", type=str, default=None, help="Single jsonl (overrides meta)")
    ap.add_argument(
        "--meta_json",
        type=str,
        default=None,
        help="recogdrive_pretrain.json — scan listed Navsim / Navsim_QA annotations",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=["Navsim_QA", "Navsim"],
        help="Keys in meta_json to include",
    )
    ap.add_argument("--max_samples", type=int, default=5000, help="Max lines read per jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    paths: List[Path] = []
    if args.jsonl_path:
        paths.append(Path(args.jsonl_path))
    elif args.meta_json:
        meta = json.loads(Path(args.meta_json).read_text(encoding="utf-8"))
        for name in args.datasets:
            block = meta.get(name)
            if not isinstance(block, dict):
                continue
            ann = block.get("annotation")
            if ann:
                paths.append(Path(ann))
    else:
        ap.error("Provide --jsonl_path or --meta_json")

    answer_lens: List[int] = []
    prompt_lens: List[int] = []

    for jp in paths:
        if not jp.is_file():
            print(f"[skip] missing file: {jp}")
            continue
        lines = list(iter_jsonl(jp, args.max_samples))
        random.Random(args.seed).shuffle(lines)
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt, answer = extract_prompt_answer(rec)
            if answer:
                answer_lens.append(len(tok.encode(answer, add_special_tokens=False)))
            if prompt:
                prompt_lens.append(len(tok.encode(prompt, add_special_tokens=False)))

    if not answer_lens:
        print("No assistant answers found; check jsonl schema.")
        return

    print("jsonl_paths:", [str(p) for p in paths])
    print("tokenizer:", args.model_path)
    print("samples_with_answer:", len(answer_lens))
    print(
        "answer_token_count: mean=%.1f std=%.1f min=%d p50=%.0f p90=%.0f p99=%.0f max=%d"
        % (
            statistics.mean(answer_lens),
            statistics.stdev(answer_lens) if len(answer_lens) > 1 else 0.0,
            min(answer_lens),
            percentile(answer_lens, 50),
            percentile(answer_lens, 90),
            percentile(answer_lens, 99),
            max(answer_lens),
        )
    )
    if prompt_lens:
        print(
            "prompt_token_count (user turn only): mean=%.1f p90=%.0f max=%d"
            % (statistics.mean(prompt_lens), percentile(prompt_lens, 90), max(prompt_lens))
        )
    print(
        "\nNote: SFT assistant strings include any CoT written in the label. "
        "`opd_max_new_tokens` should be set comfortably above high-percentile *generation* length, "
        "not necessarily equal to InternVL `max_seq_length`."
    )


if __name__ == "__main__":
    main()

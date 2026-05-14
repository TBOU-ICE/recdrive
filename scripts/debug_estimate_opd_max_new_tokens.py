#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer


def read_jsonl(path: Path, limit: int):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_answer(record):
    conv = record.get('conversations') or record.get('messages')
    if isinstance(conv, list):
        for item in conv:
            if not isinstance(item, dict):
                continue
            role = str(item.get('from', item.get('role', ''))).lower()
            if role in {'gpt', 'assistant'}:
                text = item.get('value', item.get('content', item.get('text', '')))
                if isinstance(text, str) and text.strip():
                    return text
    for key in ('answer', 'output', 'text'):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ''


def main():
    parser = argparse.ArgumentParser(description='Estimate OPD max_new_tokens from SFT meta_path datasets.')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--meta_path', required=True)
    parser.add_argument('--max_rows_per_dataset', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    with open(args.meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    lengths = []
    per_set = {}

    for name, cfg in meta.items():
        ann = cfg.get('annotation')
        if not ann:
            continue
        ann_path = Path(ann)
        if not ann_path.exists():
            print(f'[skip] {name}: annotation not found: {ann_path}')
            continue

        rows = read_jsonl(ann_path, limit=args.max_rows_per_dataset)
        if not rows:
            print(f'[skip] {name}: no valid rows')
            continue

        ds_lens = []
        for r in rows:
            ans = extract_answer(r)
            if not ans:
                continue
            n = len(tokenizer.encode(ans, add_special_tokens=False))
            ds_lens.append(n)
            lengths.append(n)

        if ds_lens:
            per_set[name] = {
                'count': len(ds_lens),
                'p95': int(np.percentile(ds_lens, 95)),
                'p99': int(np.percentile(ds_lens, 99)),
                'max': int(np.max(ds_lens)),
                'mean': float(np.mean(ds_lens)),
            }

    if not lengths:
        raise RuntimeError('No valid answer texts found in meta datasets.')

    p90 = int(np.percentile(lengths, 90))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    p995 = int(np.percentile(lengths, 99.5))
    mx = int(np.max(lengths))
    mean = float(np.mean(lengths))

    print('=== Global Answer Token Stats ===')
    print(f'count={len(lengths)} mean={mean:.2f} p90={p90} p95={p95} p99={p99} p99.5={p995} max={mx}')
    print('recommend_opd_max_new_tokens:')
    print(f'- conservative (p95 margin): {max(192, int(p95 * 1.3))}')
    print(f'- balanced (p99 margin): {max(256, int(p99 * 1.2))}')
    print(f'- aggressive (p99.5 margin): {max(320, int(p995 * 1.15))}')

    print('\n=== Per Dataset ===')
    for k, v in sorted(per_set.items()):
        print(f"{k}: count={v['count']} mean={v['mean']:.1f} p95={v['p95']} p99={v['p99']} max={v['max']}")


if __name__ == '__main__':
    main()

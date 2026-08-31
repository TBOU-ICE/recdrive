#!/usr/bin/env python3
import argparse
import json
import ast
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch


def read_first_record(jsonl_path: Path):
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Some files are python-literal-like instead of strict JSON.
                return ast.literal_eval(line)
    raise ValueError(f'No valid record found in file: {jsonl_path}')


def resolve_jsonl_path(path: Path) -> Path:
    """Allow passing either an annotation jsonl or a meta_path json."""
    if path.suffix == '.jsonl':
        return path
    if path.suffix == '.json':
        meta = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(meta, dict) or not meta:
            raise ValueError(f'Invalid meta json: {path}')
        first = next(iter(meta.values()))
        ann = first.get('annotation')
        if not ann:
            raise ValueError(f'No annotation field found in first dataset of {path}')
        return Path(ann)
    raise ValueError(f'Unsupported file type: {path}')


def extract_text(record):
    for key in ('conversations', 'messages', 'text', 'answer', 'output'):
        if key in record:
            val = record[key]
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                chunks = []
                for item in val:
                    if isinstance(item, str):
                        chunks.append(item)
                    elif isinstance(item, dict):
                        for k in ('value', 'content', 'text'):
                            if k in item and isinstance(item[k], str):
                                chunks.append(item[k])
                                break
                if chunks:
                    return '\n'.join(chunks)
    return json.dumps(record, ensure_ascii=False)


def extract_prompt_and_answer(record):
    # Common SFT schema: conversations/messages with human-assistant turns.
    conv = record.get('conversations') or record.get('messages')
    if isinstance(conv, list):
        prompt = ''
        answer = ''
        for item in conv:
            if not isinstance(item, dict):
                continue
            role = str(item.get('from', item.get('role', ''))).lower()
            text = item.get('value', item.get('content', item.get('text', '')))
            if not isinstance(text, str):
                continue
            if role in {'human', 'user'} and not prompt:
                prompt = text
            if role in {'gpt', 'assistant'}:
                answer = text
                break
        if prompt or answer:
            return prompt, answer
    raw = extract_text(record)
    return raw, ''


def main():
    parser = argparse.ArgumentParser(description='Check token length for one SFT sample.')
    parser.add_argument('--model_path', default='/mnt/models/recdrive/v1.0.0/ReCogDrive-VLM-2B')
    parser.add_argument('--jsonl_path', default='/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/pretrainingQA/Navsim_ReCogDrive/dataset_navsim_recogdrive.jsonl')
    parser.add_argument('--do_generate', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=2000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    resolved_path = resolve_jsonl_path(Path(args.jsonl_path))
    record = read_first_record(resolved_path)
    prompt, answer = extract_prompt_and_answer(record)
    answer_token_ids = tokenizer.encode(answer, add_special_tokens=False) if answer else []
    sample_text = extract_text(record)
    sample_token_ids = tokenizer.encode(sample_text, add_special_tokens=False)

    print(f'jsonl_path: {resolved_path}')
    print(f'sample_token_count: {len(sample_token_ids)}')
    print(f'gt_answer_token_count: {len(answer_token_ids)}')
    print(f'prompt_preview: {prompt[:300].replace(chr(10), " ")}')
    if answer:
        print(f'gt_answer_preview: {answer[:300].replace(chr(10), " ")}')

    if args.do_generate:
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='cuda:0' if torch.cuda.is_available() else 'cpu',
        ).eval()
        enc = tokenizer(prompt if prompt else sample_text, return_tensors='pt')
        input_ids = enc['input_ids'].to(model.device)
        attn = enc['attention_mask'].to(model.device)
        with torch.no_grad():
            out = model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        starts_with_prompt = out.shape[1] >= input_ids.shape[1] and torch.equal(out[:, :input_ids.shape[1]], input_ids)
        completion_ids = out[:, input_ids.shape[1]:] if starts_with_prompt else out
        print(f'generated_completion_token_count: {completion_ids.shape[1]}')
        print('generated_completion_preview:', tokenizer.decode(completion_ids[0], skip_special_tokens=True)[:300].replace('\n', ' '))


if __name__ == '__main__':
    main()

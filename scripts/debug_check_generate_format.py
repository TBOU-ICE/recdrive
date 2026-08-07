#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Check whether generate() returns prompt+completion or completion-only.')
    parser.add_argument('--model_path', default='/workspace/models/recdrive/v1.0.0/ReCogDrive-VLM-2B')
    parser.add_argument('--prompt', default='Hello, please output three words.')
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--use_inputs_embeds', action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map='cuda:0' if torch.cuda.is_available() else 'cpu',
    ).eval()

    enc = tokenizer(args.prompt, return_tensors='pt')
    input_ids = enc['input_ids'].to(model.device)
    attn = enc['attention_mask'].to(model.device)

    with torch.no_grad():
        if args.use_inputs_embeds:
            embeds = model.language_model.get_input_embeddings()(input_ids)
            out = model.language_model.generate(
                inputs_embeds=embeds,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            out = model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    prompt_len = input_ids.shape[1]
    out_len = out.shape[1]
    print(f'prompt_len={prompt_len}, out_len={out_len}')
    print('first_prompt_ids:', input_ids[0, :min(8, prompt_len)].tolist())
    print('first_out_ids   :', out[0, :min(8, out_len)].tolist())

    starts_with_prompt = out_len >= prompt_len and torch.equal(out[:, :prompt_len], input_ids)
    if starts_with_prompt:
        print('RESULT: prompt+completion')
        completion_ids = out[:, prompt_len:]
    else:
        print('RESULT: completion-only')
        completion_ids = out

    print('decoded_output:', tokenizer.decode(out[0], skip_special_tokens=True).replace('\n', ' '))
    print('decoded_completion:', tokenizer.decode(completion_ids[0], skip_special_tokens=True).replace('\n', ' '))


if __name__ == '__main__':
    main()

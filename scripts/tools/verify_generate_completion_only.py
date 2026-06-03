#!/usr/bin/env python3
"""
Verify whether ``model.generate(..., inputs_embeds=...)`` returns **completion-only**
token ids or **prompt + completion**.

ReCogDrive OPD expects **completion-only**: ``recogdrive_backbone.forward_with_logits`` does
``torch.cat([input_ids, generated_input_ids], dim=1)``.

This script runs a lightweight HF causal LM (default: tiny-gpt2) with ``inputs_embeds``,
then checks if the first generated column matches the last prompt token (heuristic for
full-sequence return) or differs (completion-only).

Usage:
  python scripts/tools/verify_generate_completion_only.py
  python scripts/tools/verify_generate_completion_only.py --model gpt2
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)

    with torch.no_grad():
        out_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    gen_len = out_ids.shape[1]
    prefix_match = torch.equal(out_ids[:, :prompt_len], input_ids)

    print("model:", args.model)
    print("prompt_len:", prompt_len, "generated_tensor_len:", gen_len)
    print("out_ids[:, :prompt_len] == input_ids:", bool(prefix_match))
    if prefix_match:
        completion = out_ids[:, prompt_len:]
        print("Interpretation: **prompt + completion** (full sequence).")
        print("completion_len:", completion.shape[1])
        print("completion decode:", tok.decode(completion[0], skip_special_tokens=True))
    else:
        print("Interpretation: **completion-only** (new tokens only).")
        print("full decode:", tok.decode(out_ids[0], skip_special_tokens=True))

    print(
        "\nFor ReCogDrive InternVL OPD: open `recogdrive_backbone.py` and confirm "
        "`forward_with_logits` concatenates prompt `input_ids` with `generated_input_ids`. "
        "If `generate` returned the full sequence, you would duplicate the prompt in `full_ids`."
    )


if __name__ == "__main__":
    main()

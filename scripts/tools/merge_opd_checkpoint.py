"""
Merge an OPD-trained student checkpoint (step*.pt) back into a HuggingFace
model directory so that it can be used as vlm_path in the evaluation script.

The OPD training script saves:
    {
        'step': int,
        'student_state_dict': RecogDriveBackbone.state_dict(),  # keys: model.xxx
        ...
    }

RecogDriveBackbone stores the HF InternVL model as self.model, so all keys
have a leading "model." prefix that must be stripped before loading into the
raw HF AutoModel (whose own state_dict keys are plain "xxx").

Usage:
    python scripts/tools/merge_opd_checkpoint.py \
        --base_model_path /mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/InternVL3-2B-ckpt400-merged \
        --opd_ckpt_path   /workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/train_opd_sft_8gpu_oneimg/step00002000.pt \
        --output_dir      /mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/merged_model/InternVL3-2B-opd-step2000-merged
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def merge(base_model_path: str, opd_ckpt_path: str, output_dir: str) -> None:
    print(f"[merge] Loading base model from: {base_model_path}")
    model = AutoModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    print(f"[merge] Loading OPD checkpoint from: {opd_ckpt_path}")
    load_kw = {"map_location": "cpu"}
    try:
        import inspect
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
    except Exception:
        pass
    raw = torch.load(opd_ckpt_path, **load_kw)

    if "student_state_dict" not in raw:
        raise KeyError(
            f"Expected key 'student_state_dict' in checkpoint, got: {list(raw.keys())}"
        )
    student_sd = raw["student_state_dict"]
    step = raw.get("step", "unknown")
    print(f"[merge] Checkpoint step: {step}")

    # Strip the leading "model." added by RecogDriveBackbone (self.model = AutoModel)
    base_sd = model.state_dict()
    updated, skipped_shape, skipped_missing = 0, 0, 0
    patch = {}
    for k, v in student_sd.items():
        if not k.startswith("model."):
            continue
        hf_key = k[len("model."):]
        if hf_key not in base_sd:
            skipped_missing += 1
            continue
        if base_sd[hf_key].shape != v.shape:
            print(f"[merge][warn] shape mismatch for {hf_key}: "
                  f"base={base_sd[hf_key].shape} ckpt={v.shape} — skipped")
            skipped_shape += 1
            continue
        patch[hf_key] = v
        updated += 1

    print(f"[merge] Patching {updated} tensors  "
          f"(skipped: shape_mismatch={skipped_shape}, not_in_base={skipped_missing})")
    model.load_state_dict(patch, strict=False)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] Saving merged model to: {out}")
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # InternVL uses trust_remote_code — copy custom Python files so that
    # AutoModel.from_pretrained(output_dir, trust_remote_code=True) works
    # without needing the original base_model_path to be accessible.
    import shutil
    for py_file in Path(base_model_path).glob("*.py"):
        dest = out / py_file.name
        if not dest.exists():
            shutil.copy2(py_file, dest)
            print(f"[merge] Copied custom code: {py_file.name}")

    print("[merge] Done.")


def main():
    parser = argparse.ArgumentParser(description="Merge OPD student ckpt into HF model dir")
    parser.add_argument("--base_model_path", default='/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/merged_model/InternVL3-2B-ckpt400-merged',
                        help="Path to the base HF InternVL model directory (e.g. InternVL3-2B-ckpt400-merged)")
    parser.add_argument("--opd_ckpt_path", default='/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/train_opd_sft_8gpu_oneimg/step00002000.pt',
                        help="Path to step*.pt saved by run_training_recogdrive_opd_sft.py")
    parser.add_argument("--output_dir", default='/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/merged_model/InternVL3-2B-opd-step2000-merged',
                        help="Where to save the merged HF model directory")
    args = parser.parse_args()
    merge(args.base_model_path, args.opd_ckpt_path, args.output_dir)


if __name__ == "__main__":
    main()

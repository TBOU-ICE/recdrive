#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Merge a LoRA-trained ReCogDrive VLM into a standalone full model, so it can be
# consumed by the downstream hidden-state caching / agent (which loads a plain
# InternVLChatModel via from_pretrained, not a PEFT adapter).
#
# Usage:
#   bash shell/internvl3.0/2nd_finetune/merge_simscale_lora.sh <INPUT_LORA_DIR> <OUTPUT_MERGED_DIR>
#
# Example:
#   bash shell/internvl3.0/2nd_finetune/merge_simscale_lora.sh \
#       /workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/vlm_simscale_lora \
#       /workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/vlm_simscale_lora_merged
# ---------------------------------------------------------------------------

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <INPUT_LORA_DIR> <OUTPUT_MERGED_DIR>" >&2
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNVL_CHAT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${INTERNVL_CHAT_ROOT}"

RECDRIVE_CONDA_BIN="${RECDRIVE_CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}"
export PATH="${RECDRIVE_CONDA_BIN}:${PATH}"
export PYTHONPATH="${PYTHONPATH:+"${PYTHONPATH}:"}$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-${RECDRIVE_CONDA_BIN}/python}"

if [ ! -f "${INPUT_DIR}/config.json" ]; then
  echo "[ERROR] INPUT_DIR has no config.json: ${INPUT_DIR}" >&2
  echo "        Point it at the LoRA training OUTPUT_DIR (contains trainer.save_model output)." >&2
  exit 1
fi

echo "[merge] input=${INPUT_DIR}"
echo "[merge] output=${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" tools/merge_lora.py "${INPUT_DIR}" "${OUTPUT_DIR}"

# Defensive copy: ensure custom modeling/config .py and tokenizer aux files that
# some transformers versions do not re-emit are present in the merged dir.
for f in modeling_intern_vit.py modeling_internvl_chat.py configuration_intern_vit.py \
         configuration_internvl_chat.py conversation.py preprocessor_config.json \
         tokenizer_config.json tokenizer.model vocab.json merges.txt \
         added_tokens.json special_tokens_map.json generation_config.json; do
  if [ -f "${INPUT_DIR}/${f}" ] && [ ! -f "${OUTPUT_DIR}/${f}" ]; then
    cp -a "${INPUT_DIR}/${f}" "${OUTPUT_DIR}/${f}"
    echo "[merge] copied missing ${f}"
  fi
done

echo "[merge] done. Merged full VLM at: ${OUTPUT_DIR}"
echo "[merge] use it downstream, e.g.:"
echo "  VLM_PATH=${OUTPUT_DIR} bash scripts/cache_dataset/run_caching_recogdrive_hidden_state_simscale_pdm_round0.sh"

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
#       /mnt/models/recdrive/v1.0.0/vlm_simscale_lora \
#       /mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged
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

RECDRIVE_CONDA_BIN="${RECDRIVE_CONDA_BIN:-/opt/conda/envs/recdrive/bin}"
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

# The downstream agent loads the VLM with AutoModel.from_pretrained(trust_remote_code=True),
# which REQUIRES the custom modeling/config .py referenced by config.json's auto_map.
# trainer.save_model does NOT emit those .py, so copy them from INPUT_DIR and, failing
# that, from the original base model (BASE_MODEL_PATH). Without this the merged dir
# fails to load in navsim eval/caching.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/models/recdrive/v1.0.0/ReCogDrive-VLM-2B}"
for f in modeling_intern_vit.py modeling_internvl_chat.py configuration_intern_vit.py \
         configuration_internvl_chat.py conversation.py preprocessor_config.json \
         tokenizer_config.json tokenizer.model vocab.json merges.txt \
         added_tokens.json special_tokens_map.json generation_config.json; do
  if [ -f "${OUTPUT_DIR}/${f}" ]; then
    continue
  fi
  if [ -f "${INPUT_DIR}/${f}" ]; then
    cp -a "${INPUT_DIR}/${f}" "${OUTPUT_DIR}/${f}"
    echo "[merge] copied missing ${f} (from input)"
  elif [ -n "${BASE_MODEL_PATH}" ] && [ -f "${BASE_MODEL_PATH}/${f}" ]; then
    cp -a "${BASE_MODEL_PATH}/${f}" "${OUTPUT_DIR}/${f}"
    echo "[merge] copied missing ${f} (from base)"
  fi
done

echo "[merge] done. Merged full VLM at: ${OUTPUT_DIR}"
echo "[merge] use it downstream, e.g.:"
echo "  VLM_PATH=${OUTPUT_DIR} bash scripts/cache_dataset/run_caching_recogdrive_hidden_state_simscale_pdm_round0.sh"

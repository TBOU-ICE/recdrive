#!/bin/bash
set -euo pipefail
set -x

# ---------------------------------------------------------------------------
# Incremental SFT of the ReCogDrive VLM on SimScale QA data.
#
# Continues training FROM the already-driving-pretrained ReCogDrive-VLM-2B
# (NOT raw InternVL3-2B), so the model keeps its navtrain/real-domain cognition.
#
# Two modes:
#   MODE=lora (default)  -> freeze base LLM + ViT weights, train LLM-LoRA +
#                           ViT-LoRA + MLP connector (adapts to SimScale visuals
#                           without full-param updates). Produces a LoRA ckpt ->
#                           run merge_simscale_lora.sh before hidden-state cache.
#                           Disable ViT-LoRA with USE_BACKBONE_LORA=0 if needed.
#   MODE=full            -> full-parameter finetune. Use with the *mix* meta
#                           (SimScale + navtrain replay) and a small LR to limit
#                           forgetting. Output is directly a full VLM.
#
# Usage:
#   # LoRA on SimScale-only (LLM + ViT LoRA, recommended for sim visual gap):
#   bash shell/internvl3.0/2nd_finetune/internvl3_2b_simscale_incremental.sh
#
#   # LLM-LoRA only (freeze ViT entirely):
#   USE_BACKBONE_LORA=0 bash shell/internvl3.0/2nd_finetune/internvl3_2b_simscale_incremental.sh
#
#   # Full finetune with replay:
#   MODE=full META=./shell/data_info/recogdrive_simscale_mix.json \
#     bash shell/internvl3.0/2nd_finetune/internvl3_2b_simscale_incremental.sh
#
#   # Single-GPU smoke test:
#   GPUS=1 BATCH_SIZE=1 NUM_TRAIN_EPOCHS=1 bash shell/.../internvl3_2b_simscale_incremental.sh
#
# Multi-node: set NNODES, NODE_RANK (or RANK), MASTER_ADDR, MASTER_PORT.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNVL_CHAT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${INTERNVL_CHAT_ROOT}"

_DEFAULT_RECDRIVE_BIN="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin"
_DEFAULT_TORCHRUN="${_DEFAULT_RECDRIVE_BIN}/torchrun"

internvl_sanitize_path() {
  local _p="$1" _d _out=""
  _p="${_p}:"
  while [ -n "$_p" ]; do
    _d="${_p%%:*}"
    _p="${_p#*:}"
    [ -z "$_d" ] && continue
    case "$_d" in
      *"://"*|http://*|https://*|tcp://*) continue ;;
      //*) continue ;;
      tcp) continue ;;
    esac
    [[ "$_d" =~ ^[0-9]+$ ]] && continue
    _out="${_out+${_out}:}${_d}"
  done
  printf '%s' "$_out"
}

RECDRIVE_CONDA_BIN="${RECDRIVE_CONDA_BIN:-${_DEFAULT_RECDRIVE_BIN}}"
export PATH="${RECDRIVE_CONDA_BIN}:$(internvl_sanitize_path "${PATH:-}")"
TORCHRUN="${TORCHRUN:-${_DEFAULT_TORCHRUN}}"

# ---- key knobs ----
MODE="${MODE:-lora}"                 # lora | full
# Base checkpoint to CONTINUE from: the driving-pretrained ReCogDrive VLM.
MODEL_PATH="${MODEL_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B}"
META="${META:-./shell/data_info/recogdrive_simscale_only.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/vlm_simscale_lora_vit}"

# LoRA ranks (only used when MODE=lora). Both LLM and ViT LoRA are on by default
# so the model can adapt to SimScale visual domain; set USE_BACKBONE_LORA=0 to
# freeze ViT entirely (old behavior).
USE_LLM_LORA="${USE_LLM_LORA:-16}"
USE_BACKBONE_LORA="${USE_BACKBONE_LORA:-16}"

GPUS="${GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
[ "${GRADIENT_ACC}" -lt 1 ] && GRADIENT_ACC=1
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_DYNAMIC_PATCH="${MAX_DYNAMIC_PATCH:-12}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-12288}"
# Optional hard cap on optimizer steps (overrides epochs). Handy for smoke tests.
MAX_STEPS="${MAX_STEPS:-0}"

# LR defaults differ by mode: LoRA tolerates a higher LR; full finetune uses a
# small LR to limit forgetting during incremental training.
if [ "${MODE}" = "full" ]; then
  LEARNING_RATE="${LEARNING_RATE:-1e-5}"
else
  LEARNING_RATE="${LEARNING_RATE:-4e-5}"
fi

# ---- freeze / LoRA flags per mode ----
if [ "${MODE}" = "lora" ]; then
  FREEZE_LLM=True
  FREEZE_BACKBONE=True
  FREEZE_MLP=False
  LORA_ARGS=(--use_llm_lora "${USE_LLM_LORA}")
  if [ "${USE_BACKBONE_LORA}" -gt 0 ]; then
    LORA_ARGS+=(--use_backbone_lora "${USE_BACKBONE_LORA}")
  fi
elif [ "${MODE}" = "full" ]; then
  FREEZE_LLM=False
  FREEZE_BACKBONE=False
  FREEZE_MLP=False
  LORA_ARGS=()
else
  echo "[ERROR] MODE must be 'lora' or 'full', got '${MODE}'" >&2
  exit 1
fi

# Optional extra args (e.g. hard step cap for smoke tests).
EXTRA_ARGS=()
if [ "${MAX_STEPS}" -gt 0 ]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-34239}"

export PYTHONPATH="${PYTHONPATH:+"${PYTHONPATH}:"}$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"

if command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1 && command -v x86_64-conda-linux-gnu-g++ >/dev/null 2>&1; then
  export CC="${CC:-x86_64-conda-linux-gnu-gcc}"
  export CXX="${CXX:-x86_64-conda-linux-gnu-g++}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-x86_64-conda-linux-gnu-g++}"
fi

echo "Using python: $(which python)"
echo "MODE=${MODE}  MODEL_PATH=${MODEL_PATH}"
echo "META=${META}  OUTPUT_DIR=${OUTPUT_DIR}"
echo "GPUS=${GPUS} BATCH_SIZE=${BATCH_SIZE} GRAD_ACC=${GRADIENT_ACC} LR=${LEARNING_RATE} EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "freeze_llm=${FREEZE_LLM} freeze_backbone=${FREEZE_BACKBONE} freeze_mlp=${FREEZE_MLP} lora_args=${LORA_ARGS[*]:-none}"

mkdir -p "${OUTPUT_DIR}"

# Fail fast if the base checkpoint is missing.
if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "[ERROR] MODEL_PATH does not look like an HF model dir (no config.json): ${MODEL_PATH}" >&2
  exit 1
fi

if ! python - <<'PY'
from deepspeed.ops.op_builder import FusedAdamBuilder
FusedAdamBuilder().load(verbose=False)
print('[env-check] fused_adam extension is ready')
PY
then
  echo "[env-check] fused_adam extension build failed. Check gcc/g++ and CUDA compatibility." >&2
  exit 1
fi

"${TORCHRUN}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS}" \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${MODEL_PATH}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch "${MAX_DYNAMIC_PATCH}" \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm "${FREEZE_LLM}" \
  --freeze_mlp "${FREEZE_MLP}" \
  --freeze_backbone "${FREEZE_BACKBONE}" \
  "${LORA_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --vision_select_layer -1 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --bf16 True \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 5 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

echo "=================================================="
echo "Training done. Output: ${OUTPUT_DIR}"
if [ "${MODE}" = "lora" ]; then
  echo "NEXT: merge LoRA into a full model before downstream hidden-state caching:"
  echo "  bash shell/internvl3.0/2nd_finetune/merge_simscale_lora.sh ${OUTPUT_DIR} ${OUTPUT_DIR}_merged"
  echo "Then point the caching script's VLM_PATH to ${OUTPUT_DIR}_merged"
else
  echo "NEXT: point the caching script's VLM_PATH to ${OUTPUT_DIR}"
fi

#!/bin/bash

set -euo pipefail
set -x

# Run from repo: bash shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain_nby.sh
# Or from any cwd: this file cd's to internvl_chat root below.
#
# Single-node 8 GPU (default): no extra env needed.
# Multi-node: set NNODES, NODE_RANK (or RANK), MASTER_ADDR, MASTER_PORT before launch
#   (do NOT use PyTorch WORLD_SIZE here—it is total processes, not node count).
#
# Default paths (nby / recdrive); override with env if needed:
#   TORCHRUN, RECDRIVE_CONDA_BIN, MODEL_PATH, OUTPUT_DIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNVL_CHAT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${INTERNVL_CHAT_ROOT}"

# 固定 torchrun / Python 前缀（与 recdrive conda 一致）
_DEFAULT_RECDRIVE_BIN="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin"
_DEFAULT_TORCHRUN="${_DEFAULT_RECDRIVE_BIN}/torchrun"

# 强制使用指定 conda 环境（不依赖 activate，兼容所有机器）
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
MODEL_PATH="${MODEL_PATH:-/workspace/models/recdrive/v1.0.0/InternVL3-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/models/recdrive/v1.0.0/training_recogdrive_vlm_nby}"

echo "Using python: $(which python)"
echo "Python version: $(python --version)"
echo "INTERNVL_CHAT_ROOT=${INTERNVL_CHAT_ROOT}"
echo "TORCHRUN=${TORCHRUN}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
if [ "${GRADIENT_ACC}" -lt 1 ]; then
  GRADIENT_ACC=1
fi
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

# torchrun: --nnodes = machine count; --nproc_per_node = GPUs per machine.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
RANK="${NODE_RANK}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-34229}"

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

mkdir -p "$OUTPUT_DIR"

if ! python - <<'PY'
from deepspeed.ops.op_builder import FusedAdamBuilder
FusedAdamBuilder().load(verbose=False)
print('[env-check] fused_adam extension is ready')
PY
then
  echo "[env-check] fused_adam extension build failed. Please check gcc/g++ and CUDA compatibility." >&2
  exit 1
fi

# Global batch = per_device_train_batch_size * GPUS * nnodes * gradient_accumulation_steps
# Defaults: 1 * 8 * 1 * 16 = 128

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
  --meta_path "./shell/data_info/recogdrive_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 16 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 10 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 12288 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

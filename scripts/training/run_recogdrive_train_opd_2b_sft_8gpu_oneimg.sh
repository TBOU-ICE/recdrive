#!/usr/bin/env bash
# Single-node 8-GPU run using one-image datasets only.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MASTER_PORT="${MASTER_PORT:-29501}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

META_PATH="${META_PATH:-${REPO_ROOT}/internvl_chat/shell/data_info/recogdrive_pretrain_oneimg.json}"
OUT_DIR="${OUT_DIR:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/train_opd_sft_8gpu_oneimg}"

STUDENT_PATH="${STUDENT_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/InternVL3-2B-ckpt400-merged}"
TEACHER_PATH="${TEACHER_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-8B}"

# Tuning knobs
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"          # per-GPU batch size
NUM_WORKERS="${NUM_WORKERS:-2}"        # per-rank dataloader workers
MAX_STEPS="${MAX_STEPS:-20000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
OPD_TOPK="${OPD_TOPK:-32}"
OPD_GROUP_SIZE="${OPD_GROUP_SIZE:-1}"
OPD_MAX_NEW_TOKENS="${OPD_MAX_NEW_TOKENS:-512}"
IMAGE_MAX_NUM="${IMAGE_MAX_NUM:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"        # 0 means use all valid samples
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
PRINT_EVERY="${PRINT_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
METRICS_LOG_EVERY="${METRICS_LOG_EVERY:-1}"
LR="${LR:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

_TORCHRUN_NBY="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun"
_TORCHRUN_VOL="/mnt/volumes/nby/conda_envs/recdrive/bin/torchrun"
if [ -x "$_TORCHRUN_NBY" ]; then
  TORCHRUN="$_TORCHRUN_NBY"
elif [ -x "$_TORCHRUN_VOL" ]; then
  TORCHRUN="$_TORCHRUN_VOL"
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN="torchrun"
else
  echo "error: torchrun not found (tried $_TORCHRUN_NBY, $_TORCHRUN_VOL, PATH)" >&2
  exit 2
fi

echo "[oneimg-8gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[oneimg-8gpu] META_PATH=${META_PATH} OUT_DIR=${OUT_DIR}"
echo "[oneimg-8gpu] BATCH_SIZE=${BATCH_SIZE} NUM_WORKERS=${NUM_WORKERS} OPD_GROUP_SIZE=${OPD_GROUP_SIZE} OPD_MAX_NEW_TOKENS=${OPD_MAX_NEW_TOKENS} IMAGE_MAX_NUM=${IMAGE_MAX_NUM}"

"$TORCHRUN"   --nnodes=1   --node_rank=0   --master_addr="${MASTER_ADDR}"   --master_port="${MASTER_PORT}"   --nproc_per_node="${NPROC_PER_NODE}"   "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive_opd_sft.py"   --student_model_path "${STUDENT_PATH}"   --teacher_model_path "${TEACHER_PATH}"   --meta_path "${META_PATH}"   --output_dir "${OUT_DIR}"   --batch_size "${BATCH_SIZE}"   --num_workers "${NUM_WORKERS}"   --lr "${LR}"   --weight_decay "${WEIGHT_DECAY}"   --max_steps "${MAX_STEPS}"   --warmup_steps "${WARMUP_STEPS}"   --opd_topk "${OPD_TOPK}"   --opd_group_size "${OPD_GROUP_SIZE}"   --opd_max_new_tokens "${OPD_MAX_NEW_TOKENS}"   --image_max_num "${IMAGE_MAX_NUM}"   --max_samples "${MAX_SAMPLES}"   --seed "${SEED}"   --log_every "${LOG_EVERY}"   --print_every "${PRINT_EVERY}"   --save_every "${SAVE_EVERY}"   --metrics_log_every "${METRICS_LOG_EVERY}"

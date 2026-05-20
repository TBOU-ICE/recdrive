#!/usr/bin/env bash
# Single-GPU debug: SFT-OPD distillation (arXiv:2603.25562), 1 GPU.
# Usage:
#   bash run_recogdrive_train_opd_2b_sft_1gpu_debug.sh
#   CUDA_VISIBLE_DEVICES=1 bash run_recogdrive_train_opd_2b_sft_1gpu_debug.sh
#   META_PATH=/path/to/meta.json bash run_recogdrive_train_opd_2b_sft_1gpu_debug.sh

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MASTER_PORT="${MASTER_PORT:-29501}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

META_PATH="${META_PATH:-${REPO_ROOT}/internvl_chat/shell/data_info/recogdrive_pretrain.json}"
OUT_DIR="${OUT_DIR:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/debug_opd_sft_1gpu}"

STUDENT_PATH="${STUDENT_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/InternVL3-2B-ckpt400-merged}"
TEACHER_PATH="${TEACHER_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-8B}"

# ---- torchrun resolution (same priority as other debug scripts) ----
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

echo "[debug] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  META_PATH=${META_PATH}  OUT_DIR=${OUT_DIR}"

"$TORCHRUN" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node=1 \
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive_opd_sft.py" \
  --student_model_path "${STUDENT_PATH}" \
  --teacher_model_path "${TEACHER_PATH}" \
  --meta_path "${META_PATH}" \
  --output_dir "${OUT_DIR}" \
  --batch_size 1 \
  --num_workers 0 \
  --lr 2e-6 \
  --weight_decay 1e-4 \
  --max_steps 50 \
  --warmup_steps 5 \
  --opd_topk 32 \
  --opd_group_size 1 \
  --opd_max_new_tokens 32 \
  --image_max_num 2 \
  --max_samples 200 \
  --seed 0 \
  --log_every 1 \
  --print_every 5 \
  --save_every 20

#!/usr/bin/env bash
set -euo pipefail

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/code}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS="${GPUS:-8}"

META_PATH="${META_PATH:-/workspace/code/internvl_chat/shell/data_info/recogdrive_pretrain.json}"
OUT_DIR="${OUT_DIR:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_8b_teacher_opd_2b_sft}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
  --nnodes=${NNODES} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${GPUS} \
  $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_opd_sft.py \
  --student_model_path /mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B \
  --teacher_model_path /mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-8B \
  --meta_path "${META_PATH}" \
  --output_dir "${OUT_DIR}" \
  --batch_size 1 \
  --num_workers 4 \
  --lr 2e-6 \
  --opd_topk 32 \
  --opd_group_size 4 \
  --opd_max_new_tokens 512 \
  --max_steps 20000 \
  --log_every 20 \
  --print_every 100 \
  --save_every 1000

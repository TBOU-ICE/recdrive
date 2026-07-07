#!/usr/bin/env bash
# One-GPU smoke test for mixed navtrain + SimScale GRPO/RL.
#
# Defaults to a tiny run so you can verify:
#   - cache paths resolve;
#   - union metric-cache metadata can be generated;
#   - the mixed RL reward runs;
#   - one optimizer step and one validation batch work on a single GPU.
#
# Usage:
#   bash scripts/training/run_recogdrive_train_fullmix_simscale_rl_2b_1gpu_debug.sh
#   CUDA_VISIBLE_DEVICES=3 LIMIT_TRAIN_BATCHES=4 bash scripts/training/run_recogdrive_train_fullmix_simscale_rl_2b_1gpu_debug.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-23468}"
export GPUS="${GPUS:-1}"

export MAX_EPOCHS="${MAX_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_rl_fullmix_simscale_quality_1gpu_debug}"
export MIX_ROOT="${MIX_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale/mixed_training/fullmix_simscale_rl_1gpu_debug}"

LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-2}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"

echo "[mixed-rl-1gpu-debug] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[mixed-rl-1gpu-debug] MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[mixed-rl-1gpu-debug] LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES}"
echo "[mixed-rl-1gpu-debug] LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES}"

exec bash "${SCRIPT_DIR}/run_recogdrive_train_fullmix_simscale_rl_2b.sh" \
  trainer.params.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.params.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  trainer.params.num_sanity_val_steps=0

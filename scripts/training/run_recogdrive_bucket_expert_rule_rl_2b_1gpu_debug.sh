#!/usr/bin/env bash
set -euo pipefail

# 1-GPU smoke test for rule-intersection RL:
#   40% full navtrain cache
#   30% navtrain rule_intersection bucket
#   30% SimScale quality rule_intersection bucket
#
# This wrapper keeps the same INIT_CKPT/data paths as the full launcher, but
# limits batches so it only verifies data loading, metric-cache lookup, and GRPO
# forward/backward plumbing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-23467}"
export GPUS="${GPUS:-1}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/recdrive-scene}"
export MAX_EPOCHS="${MAX_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-10}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-2}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-debug_rule_rl_40nav_30navbucket_30simbucket_1gpu}"
export SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/datasets/simscale/20260709}"
export MIX_ROOT="${MIX_ROOT:-${SIMSCALE_ROOT}/mixed_training/debug_rule_intersection_40nav_30navbucket_30simbucket}"

echo "[debug-rule-rl] GPUS=${GPUS}"
echo "[debug-rule-rl] MAX_EPOCHS=${MAX_EPOCHS}"
echo "[debug-rule-rl] BATCH_SIZE=${BATCH_SIZE}"
echo "[debug-rule-rl] LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES}"
echo "[debug-rule-rl] LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES}"
echo "[debug-rule-rl] MIX_ROOT=${MIX_ROOT}"

bash "${SCRIPT_DIR}/run_recogdrive_bucket_expert_rule_rl_2b.sh"

#!/usr/bin/env bash
set -euo pipefail

# 1-GPU smoke test for rule-intersection RL with:
#   40% navtrain full + 30% navtrain rule + 30% SimScale rule.
#
# This wrapper keeps the same data/checkpoint defaults as the 2B launcher, but
# limits batches so it is suitable for checking paths, metric cache lookup, GRPO
# forward/backward, and checkpoint loading before a full multi-GPU run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPUS="${GPUS:-1}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-23459}"

export MAX_EPOCHS="${MAX_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-10}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-2}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-debug_rule_rl_40nav_30navbucket_30simbucket_1gpu}"
export MIX_ROOT="${MIX_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale/mixed_training/debug_rule_intersection_40nav_30navbucket_30simbucket}"

# Defaults to round0 quality. Override SIM_ROUND=1 to test round1 quality.
export SIM_ROUND="${SIM_ROUND:-0}"

echo "[debug-rule-rl] GPUS=${GPUS} BATCH_SIZE=${BATCH_SIZE} MAX_EPOCHS=${MAX_EPOCHS}"
echo "[debug-rule-rl] LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES} LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES}"
echo "[debug-rule-rl] SIM_ROUND=${SIM_ROUND} MIX_ROOT=${MIX_ROOT}"

bash "${SCRIPT_DIR}/run_recogdrive_bucket_expert_rule_rl_2b.sh"

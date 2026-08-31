#!/usr/bin/env bash
# EPDMS RL expert: rule_intersection (scene_term=ec).
# Run alone on ONE 8-GPU machine (independent of the other three experts):
#   bash run_recogdrive_expert_epdms_rule.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPUS="${GPUS:-8}" \
NNODES="${NNODES:-1}" \
NODE_RANK="${NODE_RANK:-0}" \
BUCKET_NAME="rule_intersection" \
SCENE_TERM="ec" \
MASTER_PORT="${MASTER_PORT:-23483}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_expert_epdms_rl_rule_intersection}" \
exec bash run_recogdrive_expert_epdms_rl.sh "$@"

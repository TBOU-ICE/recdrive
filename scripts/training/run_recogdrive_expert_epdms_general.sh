#!/usr/bin/env bash
# EPDMS RL expert: general_or_no_tag (scene_term=none).
# Run alone on ONE 8-GPU machine (independent of the other three experts):
#   bash run_recogdrive_expert_epdms_general.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPUS="${GPUS:-8}" \
NNODES="${NNODES:-1}" \
NODE_RANK="${NODE_RANK:-0}" \
BUCKET_NAME="general_or_no_tag" \
SCENE_TERM="none" \
MASTER_PORT="${MASTER_PORT:-23481}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_expert_epdms_rl_general_or_no_tag}" \
exec bash run_recogdrive_expert_epdms_rl.sh "$@"

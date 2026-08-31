#!/usr/bin/env bash
# EPDMS RL expert: progress_curbside_stopgo (scene_term=gate).
# Run alone on ONE 8-GPU machine (independent of the other three experts):
#   bash run_recogdrive_expert_epdms_progress.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPUS="${GPUS:-8}" \
NNODES="${NNODES:-1}" \
NODE_RANK="${NODE_RANK:-0}" \
BUCKET_NAME="progress_curbside_stopgo" \
SCENE_TERM="gate" \
MASTER_PORT="${MASTER_PORT:-23482}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_expert_epdms_rl_progress_curbside_stopgo}" \
exec bash run_recogdrive_expert_epdms_rl.sh "$@"

#!/usr/bin/env bash
# EPDMS RL expert: safety_dynamics_interaction (scene_term=ep).
# Run alone on ONE 8-GPU machine (independent of the other three experts):
#   bash run_recogdrive_expert_epdms_safety.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPUS="${GPUS:-8}" \
NNODES="${NNODES:-1}" \
NODE_RANK="${NODE_RANK:-0}" \
BUCKET_NAME="safety_dynamics_interaction" \
SCENE_TERM="ep" \
MASTER_PORT="${MASTER_PORT:-23484}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_expert_epdms_rl_safety_dynamics_interaction}" \
exec bash run_recogdrive_expert_epdms_rl.sh "$@"

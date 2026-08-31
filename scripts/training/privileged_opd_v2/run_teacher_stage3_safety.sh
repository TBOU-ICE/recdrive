#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
RL_SAFETY_CKPT="${RL_SAFETY_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_safety_dynamics_interaction_rl_newvlm/2026.07.24.09.41.09/lightning_logs/version_0/checkpoints/epoch=39-step=5040.ckpt}"
need RL_SAFETY_CKPT
BUCKET_NAME="safety_dynamics_interaction" BASE_RL_CKPT="${RL_SAFETY_CKPT}" MASTER_PORT="${MASTER_PORT:-23634}" exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

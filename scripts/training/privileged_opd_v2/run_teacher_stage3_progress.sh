#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
RL_PROGRESS_CKPT="${RL_PROGRESS_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_progress_curbside_stopgo_rl_newvlm/2026.07.24.05.09.01/lightning_logs/version_0/checkpoints/epoch=38-step=6201.ckpt}"
need RL_PROGRESS_CKPT
BUCKET_NAME="progress_curbside_stopgo" BASE_RL_CKPT="${RL_PROGRESS_CKPT}" MASTER_PORT="${MASTER_PORT:-23632}" exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
RL_GENERAL_CKPT="${RL_GENERAL_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_general_or_no_tag_rl_newvlm/2026.07.24.09.07.23/lightning_logs/version_0/checkpoints/epoch=38-step=6825.ckpt}"
need RL_GENERAL_CKPT
BUCKET_NAME="general_or_no_tag" BASE_RL_CKPT="${RL_GENERAL_CKPT}" MASTER_PORT="${MASTER_PORT:-23635}" exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

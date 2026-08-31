#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need RL_SAFETY_CKPT
BUCKET_NAME="safety_dynamics_interaction" BASE_RL_CKPT="${RL_SAFETY_CKPT}" MASTER_PORT="${MASTER_PORT:-23634}" exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

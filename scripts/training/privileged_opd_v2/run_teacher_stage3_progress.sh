#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need RL_PROGRESS_CKPT
BUCKET_NAME="progress_curbside_stopgo" BASE_RL_CKPT="${RL_PROGRESS_CKPT}" MASTER_PORT="${MASTER_PORT:-23632}" exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

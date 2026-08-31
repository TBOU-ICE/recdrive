#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need RL_GENERAL_CKPT
BUCKET_NAME="general_or_no_tag" BASE_RL_CKPT="${RL_GENERAL_CKPT}" MASTER_PORT="${MASTER_PORT:-23635}" exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

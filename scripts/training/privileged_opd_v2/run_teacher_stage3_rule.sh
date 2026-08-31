#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need RL_RULE_CKPT
BUCKET_NAME="rule_intersection" BASE_RL_CKPT="${RL_RULE_CKPT}" MASTER_PORT="${MASTER_PORT:-23633}" exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

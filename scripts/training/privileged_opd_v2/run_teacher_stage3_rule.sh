#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
RL_RULE_CKPT="${RL_RULE_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_rule_intersection_rl_newvlm/2026.07.24.04.59.12/lightning_logs/version_0/checkpoints/epoch=38-step=5499.ckpt}"
need RL_RULE_CKPT
BUCKET_NAME="rule_intersection" BASE_RL_CKPT="${RL_RULE_CKPT}" MASTER_PORT="${MASTER_PORT:-23633}" exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

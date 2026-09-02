#!/usr/bin/env bash
# Goal IL teacher: T2 - rule_intersection.
# Weight-only warm-start from the 200-epoch no-goal IL teacher, then 30 epoch adaln.
#
#   SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_rule_il_goal_newvlm.sh
set -euo pipefail
export BUCKET_NAME="${BUCKET_NAME:-rule_intersection}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_rule_intersection_tokens.json}"
export MASTER_PORT="${MASTER_PORT:-23572}"
export BASE_CKPT="${BASE_CKPT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_teacher_2epoch_base_rule_intersection_il_newvlm/2026.08.31.12.20.46/lightning_logs/version_0/checkpoints/epoch=196-step=61858.ckpt}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_goal_newvlm.sh"

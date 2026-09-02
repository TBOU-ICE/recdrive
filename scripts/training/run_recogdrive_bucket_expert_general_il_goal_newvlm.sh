#!/usr/bin/env bash
# Goal IL teacher: T4 - general_or_no_tag.
# Weight-only warm-start from the 200-epoch no-goal IL teacher, then 30 epoch adaln.
#
#   SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_general_il_goal_newvlm.sh
set -euo pipefail
export BUCKET_NAME="${BUCKET_NAME:-general_or_no_tag}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_general_or_no_tag_tokens.json}"
export MASTER_PORT="${MASTER_PORT:-23574}"
export BASE_CKPT="${BASE_CKPT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_teacher_2epoch_base_general_or_no_tag_il_newvlm/2026.08.31.12.12.40/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_goal_newvlm.sh"

#!/usr/bin/env bash
# Goal IL teacher: T3 - progress_curbside_stopgo.
# Weight-only warm-start from the 200-epoch no-goal IL teacher, then 30 epoch adaln.
#
#   SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_progress_il_goal_newvlm.sh
set -euo pipefail
export BUCKET_NAME="${BUCKET_NAME:-progress_curbside_stopgo}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_progress_curbside_stopgo_tokens.json}"
export MASTER_PORT="${MASTER_PORT:-23573}"
export BASE_CKPT="${BASE_CKPT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_teacher_2epoch_base_progress_curbside_stopgo_il_newvlm/2026.08.31.12.14.02/lightning_logs/version_0/checkpoints/epoch=197-step=91872.ckpt}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_goal_newvlm.sh"

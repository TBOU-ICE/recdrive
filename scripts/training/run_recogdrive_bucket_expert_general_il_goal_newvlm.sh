#!/usr/bin/env bash
# Goal IL teacher: T4 - general_or_no_tag.
# Continue the 100-epoch adaln run (Lightning full resume: weights + optimizer
# + epoch counter). MAX_EPOCHS is the *total* epoch count, not extra epochs.
#
#   SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_general_il_goal_newvlm.sh
set -euo pipefail
export BUCKET_NAME="${BUCKET_NAME:-general_or_no_tag}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_general_or_no_tag_tokens.json}"
export MASTER_PORT="${MASTER_PORT:-23574}"
#base ckpt
export RESUME_CKPT="${RESUME_CKPT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_teacher_general_or_no_tag_200il_30goal_adaln_newvlm/2026.09.07.09.20.11/lightning_logs/version_0/checkpoints/epoch=96-step=57230.ckpt}"
export MAX_EPOCHS="${MAX_EPOCHS:-200}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_goal_newvlm.sh"

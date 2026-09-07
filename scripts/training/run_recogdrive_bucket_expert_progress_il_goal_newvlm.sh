#!/usr/bin/env bash
# Goal IL teacher: T3 - progress_curbside_stopgo.
# Weight-only warm-start from the common fullmix IL checkpoint (epoch 2), then
# 100 epoch adaln on scene-bucket-only data (navtrain bucket + SimScale rounds 0/1).
#
#   SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_progress_il_goal_newvlm.sh
set -euo pipefail
export BUCKET_NAME="${BUCKET_NAME:-progress_curbside_stopgo}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_progress_curbside_stopgo_tokens.json}"
export MASTER_PORT="${MASTER_PORT:-23573}"
export BASE_CKPT="${BASE_CKPT:-/mnt/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
export MAX_EPOCHS="${MAX_EPOCHS:-100}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_goal_newvlm.sh"

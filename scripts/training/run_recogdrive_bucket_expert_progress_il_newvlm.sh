#!/usr/bin/env bash
# STANDALONE no-goal IL teacher: T3 - progress_curbside_stopgo.
# Mixes only this bucket's navtrain + SimScale quality tokens.
# Warm-starts from the new-VLM fullmix IL ckpt (CKPT_PATH).
#
#   GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_progress_il_newvlm.sh
set -euo pipefail
export BUCKET_NAME="${BUCKET_NAME:-progress_curbside_stopgo}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_progress_curbside_stopgo_tokens.json}"
export MASTER_PORT="${MASTER_PORT:-23673}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_newvlm.sh"

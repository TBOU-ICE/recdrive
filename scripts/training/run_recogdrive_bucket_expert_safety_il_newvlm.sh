#!/usr/bin/env bash
# STANDALONE no-goal IL teacher: T1 - safety_dynamics_interaction.
# Mixes only this bucket's navtrain + SimScale quality tokens.
# Warm-starts from the new-VLM fullmix IL ckpt (CKPT_PATH).
#
#   GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_safety_il_newvlm.sh
set -euo pipefail
export BUCKET_NAME="${BUCKET_NAME:-safety_dynamics_interaction}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_safety_dynamics_interaction_tokens.json}"
export MASTER_PORT="${MASTER_PORT:-23671}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_newvlm.sh"

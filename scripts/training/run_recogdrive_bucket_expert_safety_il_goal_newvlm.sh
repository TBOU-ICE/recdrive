#!/usr/bin/env bash
# Goal IL teacher: T1 - safety_dynamics_interaction.
# Weight-only warm-start from the 200-epoch no-goal IL teacher, then 30 epoch adaln.
#
#   SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_safety_il_goal_newvlm.sh
set -euo pipefail
export BUCKET_NAME="${BUCKET_NAME:-safety_dynamics_interaction}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_safety_dynamics_interaction_tokens.json}"
export MASTER_PORT="${MASTER_PORT:-23571}"
export BASE_CKPT="${BASE_CKPT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_teacher_2epoch_base_safety_dynamics_interaction_il_newvlm/2026.08.31.12.19.19/lightning_logs/version_0/checkpoints/epoch=199-step=38800.ckpt}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_goal_newvlm.sh"

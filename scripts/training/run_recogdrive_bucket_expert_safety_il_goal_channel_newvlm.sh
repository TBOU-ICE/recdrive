#!/usr/bin/env bash
# Channel goal IL teacher: T1 - safety_dynamics_interaction.
# Fresh weight-only warm-start from the goal-free fullmix IL checkpoint.
#
#   SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_safety_il_goal_channel_newvlm.sh
set -euo pipefail
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/datasets/recdrive/20260513/exp}"
export BUCKET_NAME="${BUCKET_NAME:-safety_dynamics_interaction}"
export BUCKET_FILE="${BUCKET_FILE:-exclusive_safety_dynamics_interaction_tokens.json}"
export GOAL_MODE="${GOAL_MODE:-channel}"
export MASTER_PORT="${MASTER_PORT:-23671}"
export BASE_CKPT="${BASE_CKPT:-/mnt/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
export RESUME_CKPT="${RESUME_CKPT:-}"
export MAX_EPOCHS="${MAX_EPOCHS:-200}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_recogdrive_bucket_expert_il_goal_newvlm.sh"

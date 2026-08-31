#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Fine-tune the already-strong privileged teacher instead of relearning the
# goal branch from the goal-free IL base.  Override INIT_CKPT/LR/MAX_EPOCHS if
# you intentionally want a from-base retrain.
BUCKET_NAME="rule_intersection" \
MASTER_PORT="${MASTER_PORT:-23523}" \
INIT_CKPT="${INIT_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_rule_intersection_il_goal_adaln_newvlm/2026.08.02.16.39.57/lightning_logs/version_0/checkpoints/epoch=199-step=62800.ckpt}" \
LR="${LR:-5e-5}" \
MAX_EPOCHS="${MAX_EPOCHS:-50}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_teacher_rule_intersection_il_goal_adaln_robust_newvlm}" \
exec bash run_recogdrive_train_robust_goal_teacher_8gpu.sh "$@"

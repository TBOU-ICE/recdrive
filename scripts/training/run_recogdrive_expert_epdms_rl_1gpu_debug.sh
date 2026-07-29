#!/usr/bin/env bash
# 1-GPU debug smoke of the EPDMS RL expert training entry (tiny epoch).
#   CUDA_VISIBLE_DEVICES=1 bash run_recogdrive_expert_epdms_rl_1gpu_debug.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPUS=1 \
NNODES=1 \
NODE_RANK=0 \
BUCKET_NAME="${BUCKET_NAME:-safety_dynamics_interaction}" \
SCENE_TERM="${SCENE_TERM:-ep}" \
MASTER_PORT="${MASTER_PORT:-23499}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-3}" \
MAX_EPOCHS="${MAX_EPOCHS:-1}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
SAMPLE_TIME="${SAMPLE_TIME:-4}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-debug_expert_epdms_rl}" \
exec bash run_recogdrive_expert_epdms_rl.sh \
  trainer.params.limit_val_batches=0 \
  trainer.params.num_sanity_val_steps=0 \
  trainer.params.strategy=auto \
  "+epdms.skip_val=true" \
  "$@"

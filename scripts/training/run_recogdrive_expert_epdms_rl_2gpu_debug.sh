#!/usr/bin/env bash
# 2-GPU DDP smoke of the EPDMS RL expert training entry (tiny epoch).
#   CUDA_VISIBLE_DEVICES=0,1 bash run_recogdrive_expert_epdms_rl_2gpu_debug.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPUS=2 \
BUCKET_NAME="${BUCKET_NAME:-general_or_no_tag}" \
SCENE_TERM="${SCENE_TERM:-none}" \
MASTER_PORT="${MASTER_PORT:-23501}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-4}" \
MAX_EPOCHS="${MAX_EPOCHS:-1}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
SAMPLE_TIME="${SAMPLE_TIME:-4}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-debug_expert_epdms_rl_2gpu}" \
exec bash run_recogdrive_expert_epdms_rl.sh \
  trainer.params.limit_val_batches=0 \
  trainer.params.num_sanity_val_steps=0 \
  "+epdms.skip_val=true" \
  "$@"

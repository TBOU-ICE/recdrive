#!/usr/bin/env bash
# Plan A (DEBUG): single-GPU GRPO RL + temporal auxiliary loss.
# Mirrors run_recogdrive_train_temporal_rl_aux_2b_8gpu.sh with:
#   - 1 GPU, batch_size=2, max_epochs=1
#   - temporal_max_train_pairs=200 / val_pairs=50  (fast first-pass)
#   - HYDRA_FULL_ERROR=1 for readable stack traces
set -euo pipefail

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/code}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

MASTER_PORT="${MASTER_PORT:-23462}"

CHECKPOINT="${CHECKPOINT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_vlm_il/2026.05.19.18.10.54/lightning_logs/version_0/checkpoints/epoch=2-step=1995.ckpt}"
CACHE_PATH="${CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-debug_temporal_rl_aux_1gpu}"
LOG_ROOT="${LOG_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/debug_temporal_rl_aux_1gpu}"

TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.01}"
TEMPORAL_MIN_DT_S="${TEMPORAL_MIN_DT_S:-0.1}"
TEMPORAL_MAX_DT_S="${TEMPORAL_MAX_DT_S:-1.1}"

echo "[debug-temporal-rl-aux] CHECKPOINT=${CHECKPOINT}"
echo "[debug-temporal-rl-aux] TEMPORAL_WEIGHT=${TEMPORAL_WEIGHT}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_temporal_rl.py" \
  agent=recogdrive_agent_temporal_rl \
  agent.lr=1e-4 \
  agent.grpo=True \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  "agent.checkpoint_path='${CHECKPOINT}'" \
  "agent.reference_policy_checkpoint='${CHECKPOINT}'" \
  "agent.metric_cache_path='${METRIC_CACHE_PATH}'" \
  "agent.rl_temporal_loss_weight=${TEMPORAL_WEIGHT}" \
  agent.rl_temporal_shift_steps=1 \
  agent.rl_temporal_dt=0.5 \
  agent.rl_temporal_min_sigma=0.04 \
  trainer.params.max_epochs=1 \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes=1 \
  trainer.params.devices=1 \
  dataloader.params.batch_size=2 \
  logger.type=tensorboard \
  "+logger.save_dir='${LOG_ROOT}/tensorboard'" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=False \
  force_cache_computation=False \
  +temporal_min_dt_s="${TEMPORAL_MIN_DT_S}" \
  +temporal_max_dt_s="${TEMPORAL_MAX_DT_S}" \
  +temporal_max_train_pairs=200 \
  +temporal_max_val_pairs=50 \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

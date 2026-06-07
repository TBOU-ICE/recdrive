#!/usr/bin/env bash
# Plan A: 8-GPU GRPO RL training with temporal consistency AUXILIARY LOSS.
#
# Loss = GRPO_policy_loss(PDMS_reward) + rl_temporal_loss_weight * temporal_loss
#
# Requires the same token-level feature cache as the regular RL training.
# Unlike the regular RL script, use_cache_without_dataset must be False
# because TemporalCachePairDataset needs a SceneLoader to build adjacent-
# frame pairs from timestamp metadata.
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
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23460}"
GPUS="${GPUS:-8}"

# ── Paths (override via env vars) ────────────────────────────────────────────
# Student checkpoint: IL or previous RL DiT.
CHECKPOINT="${CHECKPOINT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_temporal_rl_loss/2026.06.04.12.33.56/checkpoints/temporal-rl-epochepoch=009-stepstep=00022000.ckpt}"
# Feature cache (same as regular RL training).
CACHE_PATH="${CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
# PDMS metric cache (same as regular RL training).
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_temporal_rl_loss_2w2_continue}"
LOG_ROOT="${LOG_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/training_temporal_rl_loss_2w2_continue}"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MAX_EPOCHS="${MAX_EPOCHS:-30}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-4}"          # pairs × 2 tokens each → effective batch = 8
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.08}"       # λ for aux temporal loss
TEMPORAL_MIN_DT_S="${TEMPORAL_MIN_DT_S:-0.1}"
TEMPORAL_MAX_DT_S="${TEMPORAL_MAX_DT_S:-1.1}"

echo "[temporal-rl-aux] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK}"
echo "[temporal-rl-aux] CHECKPOINT=${CHECKPOINT}"
echo "[temporal-rl-aux] CACHE_PATH=${CACHE_PATH}"
echo "[temporal-rl-aux] METRIC_CACHE_PATH=${METRIC_CACHE_PATH}"
echo "[temporal-rl-aux] TEMPORAL_WEIGHT=${TEMPORAL_WEIGHT}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
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
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  dataloader.params.batch_size="${PAIR_BATCH_SIZE}" \
  logger.type=tensorboard \
  "+logger.save_dir='${LOG_ROOT}/tensorboard'" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=False \
  force_cache_computation=False \
  +temporal_min_dt_s="${TEMPORAL_MIN_DT_S}" \
  +temporal_max_dt_s="${TEMPORAL_MAX_DT_S}" \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

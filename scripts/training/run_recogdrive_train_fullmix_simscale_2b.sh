#!/usr/bin/env bash
# Full-mix navtrain + SimScale DiT IL launcher.
#
# Unlike run_recogdrive_train_mixed_simscale_2b.sh (weighted ratio sampling),
# this script sets mixed_cache.fullmix=true so every cached sample has equal
# probability. The effective batch composition follows dataset size
# (navtrain ~N + simscale ~M), not a fixed 0.6/0.4 or 0.5/0.5 ratio.
#
# Does not change defaults for navtrain-only or ratio-mixed training scripts.
#
# Usage:
#   bash scripts/training/run_recogdrive_train_fullmix_simscale_2b.sh
#   GPUS=8 MAX_EPOCHS=20 bash scripts/training/run_recogdrive_train_fullmix_simscale_2b.sh

set -euo pipefail

export PATH="${CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/code}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

ROUND="${ROUND:-0}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}_quality"
SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"

NAV_CACHE_PATH="${NAV_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
SIM_CACHE_PATH="${SIM_CACHE_PATH:-${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${DATASET_NAME}}"
BASE_CKPT="${BASE_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}"
VLM_PATH="${VLM_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23457}"
GPUS="${GPUS:-8}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_il_fullmix_simscale_quality}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-5}"

#echo "[fullmix-dit] BASE_CKPT=${BASE_CKPT}"
echo "[fullmix-dit] NAV_CACHE_PATH=${NAV_CACHE_PATH}"
echo "[fullmix-dit] SIM_CACHE_PATH=${SIM_CACHE_PATH}"
echo "[fullmix-dit] mixed_cache.fullmix=true (uniform union, sample_ratios ignored)"
echo "[fullmix-dit] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  agent=recogdrive_agent \
  agent.lr="${LR}" \
  agent.grpo=False \
  agent.vlm_path="${VLM_PATH}" \
  "agent.checkpoint_path=${BASE_CKPT}" \
  agent.cam_type='single' \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  cache_path="${NAV_CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  use_mixed_cache=True \
  mixed_cache.fullmix=True \
  "mixed_cache.paths=[${NAV_CACHE_PATH},${SIM_CACHE_PATH}]" \
  'mixed_cache.names=[navtrain,simscale]' \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

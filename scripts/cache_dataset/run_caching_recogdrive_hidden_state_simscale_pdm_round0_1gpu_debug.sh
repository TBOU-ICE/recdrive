#!/usr/bin/env bash
# Single-GPU smoke test for SimScale ReCogDrive hidden-state caching.
# - nproc_per_node=1: one GPU / one process
# - train_test_split.scene_filter.max_scenes=8: only cache a few scenarios
# - CACHE_PATH uses *_debug suffix so full hidden-state cache is untouched
#
# Usage:
#   bash scripts/cache_dataset/run_caching_recogdrive_hidden_state_simscale_pdm_round0_1gpu_debug.sh
#   CUDA_VISIBLE_DEVICES=2 MAX_SCENES=4 bash scripts/cache_dataset/run_caching_recogdrive_hidden_state_simscale_pdm_round0_1gpu_debug.sh

set -euo pipefail

ROUND="${ROUND:-0}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-simscale_pdm_round0}"
MAX_SCENES="${MAX_SCENES:-8}"

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SIMSCALE_ROOT}}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${SIMSCALE_ROOT}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

CACHE_PATH="${CACHE_PATH:-${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${DATASET_NAME}_debug}"
LOG_DIR="${LOG_DIR:-${SIMSCALE_ROOT}/logs}"
CONDA_PYTHON_ROOT="${CONDA_PYTHON_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_PYTHON_ROOT}/bin/torchrun}"
VLM_PATH="${VLM_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-63669}"

export MASTER_ADDR
export MASTER_PORT

mkdir -p "${CACHE_PATH}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/caching_recogdrive_hidden_state_${DATASET_NAME}_1gpu_debug.txt"

echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"
echo "MAX_SCENES: ${MAX_SCENES}"
echo "OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
echo "NUPLAN_MAPS_ROOT: ${NUPLAN_MAPS_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "VLM_PATH: ${VLM_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

"${TORCHRUN_BIN}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py" \
    agent=recogdrive_agent \
    experiment_name="recogdrive_agent_cache_${DATASET_NAME}_1gpu_debug" \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    train_test_split.scene_filter.max_scenes="${MAX_SCENES}" \
    agent.vlm_path="${VLM_PATH}" \
    cache_path="${CACHE_PATH}" \
    force_cache_computation=true \
    worker=sequential \
    2>&1 | tee "${LOG_FILE}"

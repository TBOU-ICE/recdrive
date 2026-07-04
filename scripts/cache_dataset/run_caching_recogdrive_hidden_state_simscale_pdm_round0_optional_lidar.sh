#!/usr/bin/env bash
set -euo pipefail

ROUND="${ROUND:-0}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-simscale_pdm_round0}"

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SIMSCALE_ROOT}}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${SIMSCALE_ROOT}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

CACHE_PATH="${CACHE_PATH:-${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${DATASET_NAME}}"
LOG_DIR="${LOG_DIR:-${SIMSCALE_ROOT}/logs}"
CONDA_PYTHON_ROOT="${CONDA_PYTHON_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_PYTHON_ROOT}/bin/torchrun}"
VLM_PATH="${VLM_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-63669}"

detect_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
        local count=0
        local item
        IFS=',' read -ra visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
        for item in "${visible_devices[@]}"; do
            [[ -n "${item// /}" ]] && count=$((count + 1))
        done
        echo "${count}"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
        return
    fi

    echo 0
}

DETECTED_GPUS="$(detect_gpus)"
GPUS_PER_NODE="${GPUS:-${DETECTED_GPUS}}"

if (( GPUS_PER_NODE < 1 )); then
    echo "[ERROR] No GPU detected. This hidden-state cache needs GPU because agent.cache_hidden_state=True." >&2
    echo "        Set CUDA_VISIBLE_DEVICES or pass GPUS=<num_visible_gpus> explicitly." >&2
    exit 1
fi

export MASTER_ADDR
export MASTER_PORT

mkdir -p "${CACHE_PATH}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/caching_recogdrive_hidden_state_${DATASET_NAME}_optional_lidar_rank${NODE_RANK}.txt"

echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"
echo "OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
echo "NUPLAN_MAPS_ROOT: ${NUPLAN_MAPS_ROOT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "DETECTED_GPUS: ${DETECTED_GPUS}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "LOG_FILE: ${LOG_FILE}"

echo "[INFO] Using optional-lidar cache entrypoint; original NAVSIM files are unchanged."

"${TORCHRUN_BIN}"     --nnodes="${NNODES}"     --node_rank="${NODE_RANK}"     --master_addr="${MASTER_ADDR}"     --master_port="${MASTER_PORT}"     --nproc_per_node="${GPUS_PER_NODE}"     "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node_optional_lidar.py"     agent=recogdrive_agent     experiment_name="recogdrive_agent_cache_${DATASET_NAME}"     agent.cam_type='single'     agent.cache_hidden_state=True     agent.cache_mode=True     train_test_split="${TRAIN_TEST_SPLIT}"     agent.vlm_path="${VLM_PATH}"     cache_path="${CACHE_PATH}"     worker=sequential     2>&1 | tee "${LOG_FILE}"

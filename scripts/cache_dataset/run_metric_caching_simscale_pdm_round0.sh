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
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CACHE_PATH="${CACHE_PATH:-${SIMSCALE_ROOT}/metric_cache_${DATASET_NAME}}"
LOG_DIR="${LOG_DIR:-${SIMSCALE_ROOT}/logs}"
CONDA_PYTHON_ROOT="${CONDA_PYTHON_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PYTHON_ROOT}/bin/python}"
WORKER="${WORKER:-single_machine_thread_pool}"
METRIC_CACHE_WORKERS="${METRIC_CACHE_WORKERS:-16}"
USE_PROCESS_POOL="${USE_PROCESS_POOL:-false}"

mkdir -p "${CACHE_PATH}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/metric_caching_${DATASET_NAME}.txt"

echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"
echo "OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
echo "NUPLAN_MAPS_ROOT: ${NUPLAN_MAPS_ROOT}"
echo "WORKER: ${WORKER}"
echo "METRIC_CACHE_WORKERS: ${METRIC_CACHE_WORKERS}"
echo "USE_PROCESS_POOL: ${USE_PROCESS_POOL}"
echo "LOG_FILE: ${LOG_FILE}"

HYDRA_OVERRIDES=(
    train_test_split="${TRAIN_TEST_SPLIT}"
    cache.cache_path="${CACHE_PATH}"
    cache.force_feature_computation=false
    worker="${WORKER}"
)

if [[ "${WORKER}" == "single_machine_thread_pool" ]]; then
    HYDRA_OVERRIDES+=(
        worker.max_workers="${METRIC_CACHE_WORKERS}"
        worker.use_process_pool="${USE_PROCESS_POOL}"
    )
elif [[ "${WORKER}" == "ray_distributed" || "${WORKER}" == "ray_distributed_no_torch" ]]; then
    HYDRA_OVERRIDES+=(worker.threads_per_node="${METRIC_CACHE_WORKERS}")
fi

"${PYTHON_BIN}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
    "${HYDRA_OVERRIDES[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

#!/usr/bin/env bash
# Single-node high-parallel SimScale metric cache (CPU only, no GPU needed).
#
# Same CACHE_PATH as run_metric_caching_simscale_pdm_round0.sh — existing
# metric_cache.pkl files are skipped (resume-safe). Safe to run locally while
# another job is queued; do NOT run two jobs on the same CACHE_PATH at once.
#
# PDM-Closed is CPU-bound: use ProcessPool (default) not ThreadPool.
# Set OMP_NUM_THREADS=1 per worker to avoid oversubscribing cores.
#
# Usage:
#   bash scripts/cache_dataset/run_metric_caching_simscale_pdm_round0_local_fast.sh
#   METRIC_CACHE_WORKERS=96 bash scripts/cache_dataset/run_metric_caching_simscale_pdm_round0_local_fast.sh
#   METRIC_CACHE_WORKERS=32 USE_PROCESS_POOL=false bash ...  # thread pool (usually slower)

set -euo pipefail

ROUND="${ROUND:-0}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-simscale_pdm_round0}"

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/datasets/simscale/20260709}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SIMSCALE_ROOT}}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${SIMSCALE_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Avoid nested BLAS/OpenMP threads when running many worker processes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CACHE_PATH="${CACHE_PATH:-${SIMSCALE_ROOT}/metric_cache_${DATASET_NAME}}"
LOG_DIR="${LOG_DIR:-${SIMSCALE_ROOT}/logs}"
CONDA_PYTHON_ROOT="${CONDA_PYTHON_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PYTHON_ROOT}/bin/python}"

WORKER="${WORKER:-single_machine_thread_pool}"
USE_PROCESS_POOL="${USE_PROCESS_POOL:-true}"
FORCE_FEATURE_COMPUTATION="${FORCE_FEATURE_COMPUTATION:-false}"

_NPROC="$(nproc 2>/dev/null || echo 16)"
# Default: use most cores on this node; cap to limit shared-storage contention.
if [[ -z "${METRIC_CACHE_WORKERS:-}" ]]; then
    if ((_NPROC > 128)); then
        METRIC_CACHE_WORKERS=96
    elif ((_NPROC > 64)); then
        METRIC_CACHE_WORKERS=64
    else
        METRIC_CACHE_WORKERS="${_NPROC}"
    fi
fi

mkdir -p "${CACHE_PATH}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/metric_caching_${DATASET_NAME}_local_fast.txt"

EXISTING_CACHE_COUNT="$(find "${CACHE_PATH}" -name metric_cache.pkl 2>/dev/null | wc -l | tr -d ' ')"

echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"
echo "OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
echo "NUPLAN_MAPS_ROOT: ${NUPLAN_MAPS_ROOT}"
echo "CPU_CORES(nproc): ${_NPROC}"
echo "WORKER: ${WORKER}"
echo "METRIC_CACHE_WORKERS: ${METRIC_CACHE_WORKERS}"
echo "USE_PROCESS_POOL: ${USE_PROCESS_POOL}"
echo "FORCE_FEATURE_COMPUTATION: ${FORCE_FEATURE_COMPUTATION}"
echo "EXISTING_CACHE_COUNT: ${EXISTING_CACHE_COUNT} / 65026 (approx)"
echo "LOG_FILE: ${LOG_FILE}"
echo "NOTE: Metric caching is CPU-only; GPU will stay idle."

HYDRA_OVERRIDES=(
    train_test_split="${TRAIN_TEST_SPLIT}"
    cache.cache_path="${CACHE_PATH}"
    cache.force_feature_computation="${FORCE_FEATURE_COMPUTATION}"
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

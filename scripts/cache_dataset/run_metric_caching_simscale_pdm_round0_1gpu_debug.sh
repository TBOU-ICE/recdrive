#!/usr/bin/env bash
# Single-process smoke test for SimScale metric caching.
# - worker=sequential: no Ray / thread pool, easier to debug
# - train_test_split.scene_filter.max_scenes=8: only cache a few scenarios
# - CACHE_PATH uses *_debug suffix so full metric cache is untouched
#
# Usage:
#   bash scripts/cache_dataset/run_metric_caching_simscale_pdm_round0_1gpu_debug.sh
#   MAX_SCENES=4 bash scripts/cache_dataset/run_metric_caching_simscale_pdm_round0_1gpu_debug.sh

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
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

CACHE_PATH="${CACHE_PATH:-${SIMSCALE_ROOT}/metric_cache_${DATASET_NAME}_debug}"
LOG_DIR="${LOG_DIR:-${SIMSCALE_ROOT}/logs}"
CONDA_PYTHON_ROOT="${CONDA_PYTHON_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PYTHON_ROOT}/bin/python}"

mkdir -p "${CACHE_PATH}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/metric_caching_${DATASET_NAME}_1gpu_debug.txt"

echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"
echo "MAX_SCENES: ${MAX_SCENES}"
echo "OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
echo "NUPLAN_MAPS_ROOT: ${NUPLAN_MAPS_ROOT}"
echo "LOG_FILE: ${LOG_FILE}"

"${PYTHON_BIN}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    cache.cache_path="${CACHE_PATH}" \
    cache.force_feature_computation=true \
    train_test_split.scene_filter.max_scenes="${MAX_SCENES}" \
    worker=sequential \
    2>&1 | tee "${LOG_FILE}"

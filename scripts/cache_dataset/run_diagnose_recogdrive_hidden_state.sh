#!/usr/bin/env bash
# Diagnose ReCogDrive-VLM hidden states: navtrain (real) vs SimScale (perturbed).
# No ReCogDrive training — only distribution / classifier / perturbation checks.
#
# Usage:
#   bash scripts/cache_dataset/run_diagnose_recogdrive_hidden_state.sh
#   ROUND=1 bash scripts/cache_dataset/run_diagnose_recogdrive_hidden_state.sh
#   MAX_NAV_SAMPLES=2000 MAX_SIM_SAMPLES=2000 bash scripts/cache_dataset/run_diagnose_recogdrive_hidden_state.sh

set -euo pipefail

ROUND="${ROUND:-0}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}"
SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/datasets/simscale/20260709}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NAV_CACHE="${NAV_CACHE:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
SIM_CACHE="${SIM_CACHE:-${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SIMSCALE_ROOT}/hidden_state_diag_${DATASET_NAME}}"

CONDA_PYTHON_ROOT="${CONDA_PYTHON_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PYTHON_ROOT}/bin/python}"

MAX_NAV_SAMPLES="${MAX_NAV_SAMPLES:-4000}"
MAX_SIM_SAMPLES="${MAX_SIM_SAMPLES:-4000}"
MAX_PAIRS="${MAX_PAIRS:-3000}"
SEED="${SEED:-0}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/diagnose_hidden_state.log"

echo "NAV_CACHE: ${NAV_CACHE}"
echo "SIM_CACHE: ${SIM_CACHE}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "MAX_NAV_SAMPLES: ${MAX_NAV_SAMPLES}"
echo "MAX_SIM_SAMPLES: ${MAX_SIM_SAMPLES}"
echo "LOG_FILE: ${LOG_FILE}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/cache_dataset/diagnose_recogdrive_hidden_state.py" \
    --nav-cache "${NAV_CACHE}" \
    --sim-cache "${SIM_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --repo-root "${REPO_ROOT}" \
    --max-nav-samples "${MAX_NAV_SAMPLES}" \
    --max-sim-samples "${MAX_SIM_SAMPLES}" \
    --max-pairs "${MAX_PAIRS}" \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
PATHS_FILE="${OPD_V2_PATHS_FILE:-${SCRIPT_DIR}/paths.local.sh}"
if [[ -f "${PATHS_FILE}" ]]; then source "${PATHS_FILE}"; fi
export PATH="${CONDA_BIN:-/usr/bin}:$PATH"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

need() { local n="$1"; [[ -n "${!n:-}" ]] || { echo "[ERROR] missing env $n (see paths.example.sh)" >&2; exit 2; }; }
q() { printf "'%s'" "$1"; }

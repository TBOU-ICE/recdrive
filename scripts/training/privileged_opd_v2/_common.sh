#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
PATHS_FILE="${OPD_V2_PATHS_FILE:-${SCRIPT_DIR}/paths.local.sh}"
if [[ -f "${PATHS_FILE}" ]]; then source "${PATHS_FILE}"; fi

# Cluster defaults. Every value remains overridable by the AIJob environment or
# OPD_V2_PATHS_FILE, but the checked-in launchers do not depend on paths.local.sh.
export CONDA_BIN="${CONDA_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin}"
export VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
export NAV_CACHE="${NAV_CACHE:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
export NAV_BUCKET_ROOT="${NAV_BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain}"
export TOKEN_TO_BUCKET_NAV="${TOKEN_TO_BUCKET_NAV:-${NAV_BUCKET_ROOT}/exclusive_token_to_bucket.json}"

MANIFEST_DIR="${MANIFEST_DIR:-${NAVSIM_DEVKIT_ROOT}/data/epdms/manifests}"
if [[ ! -d "${MANIFEST_DIR}" ]]; then
  LEGACY_MANIFEST_DIR="/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1-gpt/data/epdms/manifests"
  if [[ -d "${LEGACY_MANIFEST_DIR}" ]]; then MANIFEST_DIR="${LEGACY_MANIFEST_DIR}"; fi
fi
export MANIFEST_DIR
export NAV_MANIFEST="${NAV_MANIFEST:-}"
if [[ -z "${NAV_MANIFEST}" && -f "${MANIFEST_DIR}/nav_train_newvlm.json" ]]; then
  export NAV_MANIFEST="${MANIFEST_DIR}/nav_train_newvlm.json"
fi

export SIM_CACHE_R0="${SIM_CACHE_R0:-/workspace/datasets/simscale/20260709/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0_quality}"
export SIM_CACHE_R1="${SIM_CACHE_R1:-/workspace/datasets/simscale/20260709/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-1_quality}"
export SIM_MANIFEST_R0="${SIM_MANIFEST_R0:-${MANIFEST_DIR}/sim_round0_quality_newvlm.json}"
export SIM_MANIFEST_R1="${SIM_MANIFEST_R1:-${MANIFEST_DIR}/sim_round1_quality_newvlm.json}"
export SIM_BUCKET_ROOT="${SIM_BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/simscale}"
export SIM_BUCKET_R0_ROOT="${SIM_BUCKET_R0_ROOT:-${SIM_BUCKET_ROOT}/scene_buckets_synthetic_reaction_pdm_v1.0-0_quality}"
export SIM_BUCKET_R1_ROOT="${SIM_BUCKET_R1_ROOT:-${SIM_BUCKET_ROOT}/scene_buckets_synthetic_reaction_pdm_v1.0-1_quality}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"

export PATH="${CONDA_BIN}:$PATH"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

need() { local n="$1"; [[ -n "${!n:-}" ]] || { echo "[ERROR] missing env $n (see paths.example.sh)" >&2; exit 2; }; }
need_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }; }
need_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 2; }; }
need_exec() { [[ -x "$1" ]] || { echo "[ERROR] missing executable: $1" >&2; exit 2; }; }
q() { printf "'%s'" "$1"; }

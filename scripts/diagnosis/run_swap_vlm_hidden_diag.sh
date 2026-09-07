#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-bd-su01/nby/exp}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/recdrive/bin/python}"
DIAG_ROOT="${DIAG_ROOT:-${REPO_ROOT}/exp/vlm_diag_rule_intersection}"
OUT_DIR="${OUT_DIR:-${DIAG_ROOT}/swap}"
LIMIT="${LIMIT:-20}"
TAG="${TAG:-rl_45_45}"

VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/ReCogDrive-VLM-2B}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache}"

case "${TAG}" in
  rl_45_45)
    CKPT="${CKPT:-/mnt/models/recdrive/v1.0.0/training_recogdrive_rule_rl_10nav_45navbucket_45simbucket/2026.07.13.12.11.00/lightning_logs/version_0/checkpoints/epoch=4-step=705.ckpt}"
    ;;
  rl_40_30)
    CKPT="${CKPT:-/mnt/models/recdrive/v1.0.0/training_recogdrive_rule_rl_40nav_30navbucket_30simbucket/2026.07.10.04.16.16/lightning_logs/version_0/checkpoints/epoch=13-step=1848.ckpt}"
    ;;
  il99)
    CKPT="${CKPT:-/mnt/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_round01_quality/2026.07.12.03.07.49/lightning_logs/version_0/checkpoints/epoch=99-step=156100.ckpt}"
    ;;
  fuxian)
    CKPT="${CKPT:-/mnt/models/recdrive/v1.0.0/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"
    ;;
  *)
    echo "Unknown TAG=${TAG}; set CKPT explicitly" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"
echo "TAG=${TAG} LIMIT=${LIMIT} CKPT=${CKPT}"
echo "OUT_DIR=${OUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/diagnosis/swap_vlm_hidden_diag.py" \
  --fail-tokens "${DIAG_ROOT}/fail_tokens.txt" \
  --good-tokens "${DIAG_ROOT}/good_tokens.txt" \
  --checkpoint "${CKPT}" \
  --vlm-path "${VLM_PATH}" \
  --metric-cache-path "${METRIC_CACHE_PATH}" \
  --openscene-root "${OPENSCENE_DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --limit "${LIMIT}" \
  --tag "${TAG}"

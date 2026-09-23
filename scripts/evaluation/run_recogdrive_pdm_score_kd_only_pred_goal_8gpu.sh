#!/usr/bin/env bash
# Standard navtest PDMS for the KD-only student WITH a predicted-goal head.
# Inference: observations -> predicted goal -> AdaLN-conditioned 5-step DDIM.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/opt/conda/envs/recdrive/bin}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_BIN}/torchrun}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/output/tensorboard}"

export PATH="${CONDA_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export NAVSIM_EXP_ROOT="${OUTPUT_BASE}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-63820}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
STUDENT_GOAL_MODE="${STUDENT_GOAL_MODE:-adaln}"
CHECKPOINT="${CHECKPOINT:-/mnt/datasets/recdrive/20260513/training_sft_student_opd_kd_only_v1/checkpoints/epoch=49-step=156050.ckpt}"
VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-kd-only-pred-goal-epoch49-navtest}"
MAX_SCENES="${MAX_SCENES:-}"

OUTPUT_ROOT="${OUTPUT_BASE}/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/run.log}"
mkdir -p "${OUTPUT_ROOT}"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] required file missing: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] required directory missing: $1" >&2; exit 1; }; }
require_file "${TORCHRUN_BIN}"
require_file "${CHECKPOINT}"
require_dir "${VLM_PATH}"
require_dir "${METRIC_CACHE_PATH}"
require_dir "${NUPLAN_MAPS_ROOT}"
require_dir "${OPENSCENE_DATA_ROOT}"

HYDRA_OVERRIDES=(
  train_test_split="${TRAIN_TEST_SPLIT}"
  agent=recogdrive_agent_goalbridge_eval
  agent.student_goal_mode="${STUDENT_GOAL_MODE}"
)
if [[ -n "${MAX_SCENES}" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="${MAX_SCENES}")
fi

echo "[PredGoal-Eval] checkpoint=${CHECKPOINT}"
echo "[PredGoal-Eval] goal_mode=${STUDENT_GOAL_MODE} split=${TRAIN_TEST_SPLIT} gpus=${GPUS}"
echo "[PredGoal-Eval] output=${OUTPUT_ROOT}"

"${TORCHRUN_BIN}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive_goalbridge.py" \
  "${HYDRA_OVERRIDES[@]}" \
  "agent.checkpoint_path='${CHECKPOINT}'" \
  "agent.vlm_path='${VLM_PATH}'" \
  agent.cam_type=single \
  agent.grpo=false \
  agent.cache_hidden_state=false \
  agent.cache_mode=false \
  agent.vlm_type=internvl \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  "metric_cache_path='${METRIC_CACHE_PATH}'" \
  "agent.metric_cache_path='${METRIC_CACHE_PATH}'" \
  experiment_name="${EXPERIMENT_NAME}" \
  worker=sequential 2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash
# Deployable GoalBridge student PDMS (navtest): predicted goal + DiT, no GT goal.
#
# Isolated from plain DiT PDMS and privileged teacher *goal.sh evals.
#
# Usage:
#   bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goalbridge.sh
#   CHECKPOINT=/path/to.ckpt bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goalbridge.sh
#   MAX_SCENES=8 bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goalbridge.sh
#   GOAL_SOURCE=off MAX_SCENES=64 bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goalbridge.sh

set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CONDA_BIN="${CONDA_BIN:-/opt/conda/envs/recdrive/bin}"
export PATH="${CONDA_BIN}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT="${MASTER_PORT:-63743}"
PORT="${PORT:-63742}"
export MASTER_PORT PORT

GPUS="${GPUS:-8}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"

STUDENT_GOAL_MODE="${STUDENT_GOAL_MODE:-adaln}"
STUDENT_ADALN_BOUND="${STUDENT_ADALN_BOUND:-8.0}"
GOAL_SOURCE="${GOAL_SOURCE:-pred}"
CHECKPOINT="${CHECKPOINT:-/mnt/datasets/recdrive/20260513/training_goalbridge_opd_channel_teacher_v1/checkpoints/epoch=40-step=127961.ckpt}"
VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache}"
MAX_SCENES="${MAX_SCENES:-}"
EVAL_SEED="${EVAL_SEED:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-pdms-opd-v1-channel-teacher-navtest}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_BIN}/torchrun}"

[[ "${GOAL_SOURCE}" == "pred" || "${GOAL_SOURCE}" == "off" ]] || {
  echo "[ERROR] GOAL_SOURCE must be 'pred' or 'off', got: ${GOAL_SOURCE}" >&2
  exit 1
}
[[ -x "${TORCHRUN_BIN}" ]] || { echo "[ERROR] torchrun missing: ${TORCHRUN_BIN}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] checkpoint missing: ${CHECKPOINT}" >&2; exit 1; }
[[ -d "${VLM_PATH}" ]] || { echo "[ERROR] VLM path missing: ${VLM_PATH}" >&2; exit 1; }
[[ -d "${METRIC_CACHE_PATH}" ]] || { echo "[ERROR] metric cache missing: ${METRIC_CACHE_PATH}" >&2; exit 1; }

HYDRA_OVERRIDES=(
  train_test_split="${TRAIN_TEST_SPLIT}"
  agent=recogdrive_agent_goalbridge_eval
  agent.student_goal_mode="${STUDENT_GOAL_MODE}"
  agent.student_adaln_bound="${STUDENT_ADALN_BOUND}"
  agent.goal_source="${GOAL_SOURCE}"
  +seed="${EVAL_SEED}"
)
if [[ -n "${MAX_SCENES}" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="${MAX_SCENES}")
fi

"${TORCHRUN_BIN}" \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive_goalbridge.py" \
    "${HYDRA_OVERRIDES[@]}" \
    "agent.checkpoint_path=\"${CHECKPOINT}\"" \
    "agent.vlm_path=\"${VLM_PATH}\"" \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_mode=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name="${EXPERIMENT_NAME}" \
    worker=sequential

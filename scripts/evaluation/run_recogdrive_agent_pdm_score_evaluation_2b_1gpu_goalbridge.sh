#!/usr/bin/env bash
# Deployable GoalBridge student PDMS (navtest): predicted goal + DiT, no GT goal. 8-GPU.
#
# Isolated from plain DiT PDMS and privileged teacher *goal.sh evals.
#
# Usage:
#   bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goalbridge.sh
#   CHECKPOINT=/path/to.ckpt bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goalbridge.sh
#   MAX_SCENES=8 bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goalbridge.sh

set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export PATH="/opt/conda/envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT="${MASTER_PORT:-63789}"
PORT="${PORT:-63788}"
export MASTER_PORT PORT

GPUS="${GPUS:-8}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"

# Scheme A: teachers are cross + GT goal; the deployable student is adaln + predicted goal.
STUDENT_GOAL_MODE="${STUDENT_GOAL_MODE:-adaln}"
CHECKPOINT="${CHECKPOINT:-/mnt/datasets/recdrive/20260513/training_goalbridge_opd_channel_teacher_v1/checkpoints/epoch=49-step=156050.ckpt}"
VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache}"
MAX_SCENES="${MAX_SCENES:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-pdms-channel-opd-v1-${STUDENT_GOAL_MODE}-navtest-49epoch}"

HYDRA_OVERRIDES=(
  train_test_split="${TRAIN_TEST_SPLIT}"
  agent=recogdrive_agent_goalbridge_eval
  agent.student_goal_mode="${STUDENT_GOAL_MODE}"
)
if [[ -n "${MAX_SCENES}" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="${MAX_SCENES}")
fi

/opt/conda/envs/recdrive/bin/torchrun \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive_goalbridge.py" \
    "${HYDRA_OVERRIDES[@]}" \
    "agent.checkpoint_path=\"${CHECKPOINT}\"" \
    "agent.vlm_path=\"${VLM_PATH}\"" \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name="${EXPERIMENT_NAME}" \
    worker=sequential

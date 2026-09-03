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

export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# PDMS scoring has no collectives until the final all_gather. The default
# 600s NCCL watchdog kills the first finished rank while rank 0 is still
# scoring. Disable the abort and wait up to 2h for stragglers.
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"

MASTER_PORT="${MASTER_PORT:-63780}"
PORT="${PORT:-63781}"
export MASTER_PORT PORT

GPUS="${GPUS:-8}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"

STUDENT_GOAL_MODE="${STUDENT_GOAL_MODE:-adaln}"
CHECKPOINT="${CHECKPOINT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_hopd_old_teacher_opd_gpt_v3/2026.09.01.15.53.55/lightning_logs/version_0/checkpoints/ckpt/epoch=9-step=28360.ckpt}"
VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache}"
MAX_SCENES="${MAX_SCENES:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-pdms-${STUDENT_GOAL_MODE}-hopd-gpt-v3-test1-9epoch}"

HYDRA_OVERRIDES=(
  train_test_split="${TRAIN_TEST_SPLIT}"
  agent=recogdrive_agent_goalbridge_eval
  agent.student_goal_mode="${STUDENT_GOAL_MODE}"
)
if [[ -n "${MAX_SCENES}" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="${MAX_SCENES}")
fi

/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
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

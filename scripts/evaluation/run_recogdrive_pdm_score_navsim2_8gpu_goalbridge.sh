#!/usr/bin/env bash
# Deployable GoalBridge student one-stage ePDMS (navsim_v2 / navtest).
# Predicted goal + DiT, no GT goal. 8-GPU.
#
# Isolated from run_recogdrive_pdm_score_one_stage_evaluation.sh (plain DiT)
# and *by_scene_goal.sh (privileged GT-goal teacher).
#
# Usage:
#   bash scripts/evaluation/run_recogdrive_pdm_score_navsim2_8gpu_goalbridge.sh
#   CHECKPOINT=/path/to.ckpt bash scripts/evaluation/run_recogdrive_pdm_score_navsim2_8gpu_goalbridge.sh
#   MAX_SCENES=8 bash scripts/evaluation/run_recogdrive_pdm_score_navsim2_8gpu_goalbridge.sh

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:${PATH}"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/navsim_v2}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
STUDENT_GOAL_MODE="${STUDENT_GOAL_MODE:-adaln}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-navsim2-goalbridge-predgoal-${STUDENT_GOAL_MODE}-epoch49}"
MASTER_PORT="${MASTER_PORT:-63790}"
MAX_SCENES="${MAX_SCENES:-}"

VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
VLM_WEIGHTS_PATH="${VLM_WEIGHTS_PATH:-}"
CHECKPOINT="${CHECKPOINT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_goalbridge_opd_old_teacher_gpt_rl_anchor_v1/2026.08.26.21.57.12/lightning_logs/version_0/checkpoints/epoch=49-step=141800.ckpt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_v2}"

TORCHRUN="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"

HYDRA_OVERRIDES=(
  train_test_split="${TRAIN_TEST_SPLIT}"
  agent=recogdrive_agent_goalbridge_eval
  agent.student_goal_mode="${STUDENT_GOAL_MODE}"
)
if [[ -n "${MAX_SCENES}" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="${MAX_SCENES}")
fi

"${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_goalbridge.py" \
  "${HYDRA_OVERRIDES[@]}" \
  experiment_name="${EXPERIMENT_NAME}" \
  "agent.checkpoint_path='${CHECKPOINT}'" \
  "agent.vlm_path='${VLM_PATH}'" \
  "agent.vlm_weights_path='${VLM_WEIGHTS_PATH}'" \
  agent.cam_type=single \
  agent.grpo=False \
  agent.cache_hidden_state=False \
  agent.cache_mode=False \
  agent.vlm_type=internvl \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  "agent.metric_cache_path='${METRIC_CACHE_PATH}'" \
  worker=sequential

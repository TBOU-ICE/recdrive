#!/usr/bin/env bash
set -x

# Single-GPU full-navtest PDMS eval for a GOAL-CONDITIONED 2B agent.
# Counterpart of run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_debug.sh, but:
#   * train_test_split=navtest  (full scene, not a bucket)
#   * run_pdm_score_recogdrive_goal.py + agent=recogdrive_goal_agent
#   * privileged GT goal point is fed at eval (oracle goal)
#
# Usage:
#   bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goal.sh
#   GOAL_MODE=cross CHECKPOINT=/path/to.ckpt \
#     bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goal.sh
#   # quick smoke:
#   MAX_SCENES=8 bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_goal.sh

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-scene}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63670}
PORT=${PORT:-63671}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# GOAL_MODE must match the mode the checkpoint was TRAINED with (adaln / channel /
# cross).  A mismatch does not crash -- the checkpoint is loaded with strict=False,
# mode-specific projection weights are silently dropped, and the goal encoder feeds
# a pathway it was never trained for -- which wrecks the score.
GOAL_MODE="${GOAL_MODE:-channel}"
CHECKPOINT="${CHECKPOINT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_goal_channel_newvlm/2026.08.03.14.51.54/lightning_logs/version_0/checkpoints/epoch=47-step=74928.ckpt}"

# PDMS on navtest must use caches built for that split; metric_cache_train tokens won't match.
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache}"
MAX_SCENES="${MAX_SCENES:-}"

HYDRA_OVERRIDES=(
  train_test_split="${TRAIN_TEST_SPLIT}"
  agent=recogdrive_goal_agent
  agent.goal_mode="${GOAL_MODE}"
)
if [[ -n "${MAX_SCENES}" ]]; then
  HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="${MAX_SCENES}")
fi

/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive_goal.py" \
    "${HYDRA_OVERRIDES[@]}" \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path='/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name="${EXPERIMENT_NAME:-eval-pdms-fullmix-goal-${GOAL_MODE}-navtest}" \
    worker=sequential

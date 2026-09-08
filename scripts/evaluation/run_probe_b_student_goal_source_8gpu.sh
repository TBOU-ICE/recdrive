#!/usr/bin/env bash
# Probe B: how much of the GoalBridge student's gap is goal prediction?
#
# Same student checkpoint, same split, same everything -- only the goal source
# changes:
#   pred : the deployable number you already have (~86)
#   gt   : ORACLE goal from the Scene (report as such, it is not deployable)
#
# Read the result as:
#   gt ~= teacher score  -> distillation transfer is lossless, the whole gap is
#                           the goal head. Self-distillation will not help.
#   gt <  teacher score  -> that shortfall is teacher/student mismatch, which is
#                           exactly what privileged self-distillation removes.
#
# Usage:
#   bash scripts/evaluation/run_probe_b_student_goal_source_8gpu.sh
#   GOAL_SOURCES="gt" MAX_SCENES=64 bash scripts/evaluation/run_probe_b_student_goal_source_8gpu.sh
#   GOAL_SOURCES="gt_noisy" GOAL_NOISE_STD=1.6 bash scripts/evaluation/run_probe_b_student_goal_source_8gpu.sh

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

# Probe A scores one bucket-specific teacher per navtest_<bucket> split, so the
# student must be scored on those same splits for the 2x2 to be comparable.
# Set BUCKETS="" to score the whole navtest split instead.
BUCKETS="${BUCKETS-rule_intersection progress_curbside_stopgo safety_dynamics_interaction general_or_no_tag}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-63791}"
MAX_SCENES="${MAX_SCENES:-}"

if [[ -z "${BUCKETS}" ]]; then
  SPLITS="${TRAIN_TEST_SPLIT:-navtest}"
else
  SPLITS=""
  for b in ${BUCKETS}; do SPLITS="${SPLITS} navtest_${b}"; done
fi

STUDENT_GOAL_MODE="${STUDENT_GOAL_MODE:-adaln}"
GOAL_SOURCES="${GOAL_SOURCES:-pred gt}"
GOAL_NOISE_STD="${GOAL_NOISE_STD:-0.0}"

VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
VLM_WEIGHTS_PATH="${VLM_WEIGHTS_PATH:-}"
CHECKPOINT="${CHECKPOINT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_goalbridge_opd_old_teacher_gpt_rl_anchor_v1/2026.08.26.21.57.12/lightning_logs/version_0/checkpoints/epoch=49-step=141800.ckpt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_v2}"

TORCHRUN="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun"

for SPLIT in ${SPLITS}; do
  for GOAL_SOURCE in ${GOAL_SOURCES}; do
    echo "========== probe B: ${SPLIT} / goal_source=${GOAL_SOURCE} =========="
    EXP="${EXPERIMENT_PREFIX:-probeB-student}-${SPLIT#navtest_}-${GOAL_SOURCE}"
    if [[ "${GOAL_SOURCE}" == "gt_noisy" ]]; then
      EXP="${EXP}-std${GOAL_NOISE_STD}"
    fi

    HYDRA_OVERRIDES=(
      train_test_split="${SPLIT}"
      agent=recogdrive_agent_goalbridge_probe
      agent.student_goal_mode="${STUDENT_GOAL_MODE}"
      agent.goal_source="${GOAL_SOURCE}"
      agent.goal_noise_std="${GOAL_NOISE_STD}"
    )
    if [[ -n "${MAX_SCENES}" ]]; then
      HYDRA_OVERRIDES+=(train_test_split.scene_filter.max_scenes="${MAX_SCENES}")
    fi

    "${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
      "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py" \
      "${HYDRA_OVERRIDES[@]}" \
      experiment_name="${EXP}" \
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
  done
done

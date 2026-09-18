#!/usr/bin/env bash
# Probe A: what is the privileged teacher actually worth under realistic goal error?
#
# Per scene bucket, the bucket's own teacher checkpoint is scored twice on the
# same split:
#   gt           : the published privileged number (~94) -- upper bound with a leaked goal
#   student_pred : the exact goal the deployed GoalBridge student would produce
#
# If student_pred lands near the student's own score (~86), the student has
# already absorbed the teacher and 86 is the ceiling for this goal head; changing
# the distillation objective cannot help. The gap between gt and student_pred is
# the price of the privileged information, i.e. Delta(g).
#
# GOAL_SOURCES="gt_noisy" with a NOISE_STDS sweep traces the whole
# PDMS-vs-goal-error curve, which is cheaper than it looks and more informative
# than a single point.
#
# Usage:
#   bash scripts/evaluation/run_probe_a_teacher_goal_source.sh
#   BUCKETS=rule_intersection bash scripts/evaluation/run_probe_a_teacher_goal_source.sh
#   GOAL_SOURCES=gt_noisy NOISE_STDS="0.4 0.8 1.6 2.4 3.2" bash scripts/evaluation/run_probe_a_teacher_goal_source.sh

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

BUCKETS="${BUCKETS:-rule_intersection progress_curbside_stopgo safety_dynamics_interaction general_or_no_tag}"
GOAL_SOURCES="${GOAL_SOURCES:-gt student_pred}"
NOISE_STDS="${NOISE_STDS:-0.0}"
GOAL_MODE="${GOAL_MODE:-adaln}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-63792}"
MAX_SCENES="${MAX_SCENES:-}"

TEACHER_ROOT="${TEACHER_ROOT:-/workspace/models/recdrive/v1.0.0}"
TEACHER_CKPT_rule_intersection="${TEACHER_CKPT_rule_intersection:-${TEACHER_ROOT}/training_teacher_rule_intersection_il_goal_adaln_newvlm/2026.08.02.16.39.57/lightning_logs/version_0/checkpoints/epoch=199-step=62800.ckpt}"
TEACHER_CKPT_progress_curbside_stopgo="${TEACHER_CKPT_progress_curbside_stopgo:-${TEACHER_ROOT}/training_teacher_progress_curbside_stopgo_il_goal_adaln_newvlm/2026.08.04.10.54.30/lightning_logs/version_0/checkpoints/epoch=199-step=92800.ckpt}"
TEACHER_CKPT_safety_dynamics_interaction="${TEACHER_CKPT_safety_dynamics_interaction:-${TEACHER_ROOT}/training_teacher_safety_dynamics_interaction_il_goal_adaln_newvlm/2026.08.04.10.25.26/lightning_logs/version_0/checkpoints/epoch=199-step=38800.ckpt}"
TEACHER_CKPT_general_or_no_tag="${TEACHER_CKPT_general_or_no_tag:-${TEACHER_ROOT}/training_teacher_general_or_no_tag_il_goal_adaln_newvlm/2026.08.04.10.44.01/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"

STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_goalbridge_opd_old_teacher_gpt_rl_anchor_v1/2026.08.26.21.57.12/lightning_logs/version_0/checkpoints/epoch=49-step=141800.ckpt}"
STUDENT_GOAL_MODE="${STUDENT_GOAL_MODE:-adaln}"

VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
VLM_WEIGHTS_PATH="${VLM_WEIGHTS_PATH:-}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_v2}"

TORCHRUN="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun"

run_one() {
  local bucket="$1" goal_source="$2" noise_std="$3" teacher_ckpt="$4"
  local exp="${EXPERIMENT_PREFIX:-probeA-teacher}-${bucket}-${goal_source}"
  if [[ "${goal_source}" == "gt_noisy" ]]; then
    exp="${exp}-std${noise_std}"
  fi

  local overrides=(
    train_test_split="navtest_${bucket}"
    agent=recogdrive_goal_teacher_probe
    agent.goal_mode="${GOAL_MODE}"
    agent.goal_source="${goal_source}"
    agent.goal_noise_std="${noise_std}"
    agent.student_goal_mode="${STUDENT_GOAL_MODE}"
  )
  if [[ -n "${MAX_SCENES}" ]]; then
    overrides+=(train_test_split.scene_filter.max_scenes="${MAX_SCENES}")
  fi

  "${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py" \
    "${overrides[@]}" \
    experiment_name="${exp}" \
    "agent.checkpoint_path='${teacher_ckpt}'" \
    "agent.student_checkpoint_path='${STUDENT_CHECKPOINT}'" \
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
}

for BUCKET in ${BUCKETS}; do
  eval "TEACHER_CKPT=\${TEACHER_CKPT_${BUCKET}}"
  if [[ ! -e "${TEACHER_CKPT}" ]]; then
    echo "!! missing teacher checkpoint for ${BUCKET}: ${TEACHER_CKPT}" >&2
    exit 1
  fi
  for GOAL_SOURCE in ${GOAL_SOURCES}; do
    if [[ "${GOAL_SOURCE}" == "gt_noisy" ]]; then
      for STD in ${NOISE_STDS}; do
        echo "========== probe A: ${BUCKET} / ${GOAL_SOURCE} std=${STD} =========="
        run_one "${BUCKET}" "${GOAL_SOURCE}" "${STD}" "${TEACHER_CKPT}"
      done
    else
      echo "========== probe A: ${BUCKET} / ${GOAL_SOURCE} =========="
      run_one "${BUCKET}" "${GOAL_SOURCE}" "0.0" "${TEACHER_CKPT}"
    fi
  done
done

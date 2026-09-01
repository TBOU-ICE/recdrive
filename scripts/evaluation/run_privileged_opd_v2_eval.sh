#!/usr/bin/env bash
set -euo pipefail

# Unified PDMS / NAVSIM v2 EPDMS / NavHard evaluator for Stage-3 teachers.
# SCENE accepts progress, rule, safety, general, or all.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAINING_PATHS="${OPD_V2_PATHS_FILE:-${REPO_ROOT}/scripts/training/privileged_opd_v2/paths.local.sh}"
if [[ -f "${TRAINING_PATHS}" ]]; then
  # shellcheck source=/dev/null
  source "${TRAINING_PATHS}"
fi

export PRIVILEGED_OPD_V2_ROOT="${PRIVILEGED_OPD_V2_ROOT:-${REPO_ROOT}}"
export NAVSIM_V2_ROOT="${NAVSIM_V2_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/navsim_v2}"
export CONDA_BIN="${CONDA_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin}"
export VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
export VLM_WEIGHTS_PATH="${VLM_WEIGHTS_PATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"
export PDMS_METRIC_CACHE="${PDMS_METRIC_CACHE:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache}"
export EPDMS_METRIC_CACHE="${EPDMS_METRIC_CACHE:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_v2}"
export NAVHARD_METRIC_CACHE="${NAVHARD_METRIC_CACHE:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_navhard}"
export NAVHARD_SENSOR_PATH="${NAVHARD_SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/sensor_blobs}"
export NAVHARD_SCENES_PATH="${NAVHARD_SCENES_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles}"

METRIC="${METRIC:-pdms}"
SCENE="${SCENE:-all}"
GOAL_ON="${GOAL_ON:-1}"
GOAL_INJECTION="${GOAL_INJECTION:-gated_cross}"
GOAL_POINT_MODE="${GOAL_POINT_MODE:-multi3}"
GOAL_INDICES="${GOAL_INDICES:-[1,4,7]}"
GOAL_SINCOS_DIM="${GOAL_SINCOS_DIM:-128}"
GOAL_HIDDEN_DIM="${GOAL_HIDDEN_DIM:-512}"
GOAL_USE_HEADING="${GOAL_USE_HEADING:-false}"
GOAL_ADAPTER_HEADS="${GOAL_ADAPTER_HEADS:-8}"
PAIRED_EVAL_NOISE="${PAIRED_EVAL_NOISE:-true}"
EVAL_SEED="${EVAL_SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-$((62000 + RANDOM % 3000))}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_SCENES="${MAX_SCENES:-}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-eval-privileged-opd-v2-${METRIC}-goal${GOAL_ON}}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PATH="${CONDA_BIN}:${PATH}"

die() { echo "[ERROR] $*" >&2; exit 2; }
need_file() { [[ -f "$1" ]] || die "missing file: $1"; }
need_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
need_exec() { [[ -x "$1" ]] || die "missing executable: $1"; }

case "${METRIC}" in
  pdms|epdms|navhard) ;;
  *) die "METRIC must be pdms, epdms, or navhard" ;;
esac
case "${GOAL_ON}" in
  0) PRIVILEGED_GOAL=false ;;
  1) PRIVILEGED_GOAL=true ;;
  *) die "GOAL_ON must be 0 or 1" ;;
esac
case "${PAIRED_EVAL_NOISE}" in
  true|false) ;;
  *) die "PAIRED_EVAL_NOISE must be true or false" ;;
esac
case "${GOAL_INJECTION}" in
  adaln|cross|gated_cross) ;;
  *) die "GOAL_INJECTION must be adaln, cross, or gated_cross" ;;
esac
case "${GOAL_POINT_MODE}" in
  final|multi3) ;;
  *) die "GOAL_POINT_MODE must be final or multi3" ;;
esac

normalize_scene() {
  case "$1" in
    progress|progress_curbside_stopgo) echo progress_curbside_stopgo ;;
    rule|rule_intersection) echo rule_intersection ;;
    safety|safety_dynamics_interaction) echo safety_dynamics_interaction ;;
    general|general_or_no_tag) echo general_or_no_tag ;;
    all) echo all ;;
    *) die "unknown SCENE '$1'; use progress, rule, safety, general, or all" ;;
  esac
}

checkpoint_for_bucket() {
  local bucket="$1"
  if [[ "${SCENE}" != "all" && -n "${CHECKPOINT:-}" ]]; then
    echo "${CHECKPOINT}"
    return
  fi
  case "${bucket}" in
    progress_curbside_stopgo) echo "${PRIV_PROGRESS_CKPT:-}" ;;
    rule_intersection) echo "${PRIV_RULE_CKPT:-}" ;;
    safety_dynamics_interaction) echo "${PRIV_SAFETY_CKPT:-}" ;;
    general_or_no_tag) echo "${PRIV_GENERAL_CKPT:-}" ;;
  esac
}

SCENE="$(normalize_scene "${SCENE}")"
ALL_BUCKETS=(
  progress_curbside_stopgo
  rule_intersection
  safety_dynamics_interaction
  general_or_no_tag
)
if [[ "${SCENE}" == "all" ]]; then
  BUCKETS=("${ALL_BUCKETS[@]}")
else
  BUCKETS=("${SCENE}")
fi

TORCHRUN="${CONDA_BIN}/torchrun"
need_exec "${TORCHRUN}"
need_dir "${VLM_PATH}"
need_dir "${NUPLAN_MAPS_ROOT}"
need_dir "${OPENSCENE_DATA_ROOT}"
need_file "${REPO_ROOT}/privileged_opd_v2_eval_agent.py"
need_file "${REPO_ROOT}/navsim/agents/recogdrive/privileged_opd_v2/goal_adapter_planner.py"

case "${METRIC}" in
  pdms)
    RUNNER="${REPO_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py"
    METRIC_CACHE="${PDMS_METRIC_CACHE}"
    ;;
  epdms)
    RUNNER="${NAVSIM_V2_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py"
    METRIC_CACHE="${EPDMS_METRIC_CACHE}"
    ;;
  navhard)
    RUNNER="${NAVSIM_V2_ROOT}/navsim/planning/script/run_pdm_score.py"
    METRIC_CACHE="${NAVHARD_METRIC_CACHE}"
    need_dir "${NAVHARD_SENSOR_PATH}"
    need_dir "${NAVHARD_SCENES_PATH}"
    ;;
esac
need_file "${RUNNER}"
need_dir "${METRIC_CACHE}"

for bucket in "${BUCKETS[@]}"; do
  checkpoint="$(checkpoint_for_bucket "${bucket}")"
  [[ -n "${checkpoint}" ]] || die "missing PRIV_*_CKPT for ${bucket}"
  need_file "${checkpoint}"
done

echo "[eval] metric=${METRIC} scene=${SCENE} goal=${PRIVILEGED_GOAL}"
echo "[eval] architecture=${GOAL_POINT_MODE}+${GOAL_INJECTION} nproc=${NPROC_PER_NODE}"
echo "[eval] paired_noise=${PAIRED_EVAL_NOISE} eval_seed=${EVAL_SEED}"
echo "[eval] metric_cache=${METRIC_CACHE}"
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[eval] preflight passed"
  exit 0
fi

for index in "${!BUCKETS[@]}"; do
  bucket="${BUCKETS[$index]}"
  checkpoint="$(checkpoint_for_bucket "${bucket}")"
  port=$((MASTER_PORT + index))
  if [[ "${METRIC}" == "navhard" ]]; then
    split="navhard_${bucket}_two_stage"
  else
    split="navtest_${bucket}"
  fi

  agent_overrides=(
    agent=recogdrive_agent
    agent._target_=privileged_opd_v2_eval_agent.ReCogDrivePrivilegedGoalAdapterEvalAgent
    "+agent.privileged_goal=${PRIVILEGED_GOAL}"
    "+agent.strict_eval_goal=true"
    "+agent.goal_injection=${GOAL_INJECTION}"
    "+agent.goal_point_mode=${GOAL_POINT_MODE}"
    "+agent.goal_indices=${GOAL_INDICES}"
    "+agent.goal_sincos_dim=${GOAL_SINCOS_DIM}"
    "+agent.goal_hidden_dim=${GOAL_HIDDEN_DIM}"
    "+agent.goal_use_heading=${GOAL_USE_HEADING}"
    "+agent.goal_adapter_heads=${GOAL_ADAPTER_HEADS}"
    "+agent.paired_eval_noise=${PAIRED_EVAL_NOISE}"
    "+agent.eval_seed=${EVAL_SEED}"
    "agent.checkpoint_path='${checkpoint}'"
    "agent.vlm_path=${VLM_PATH}"
    agent.cam_type=single
    agent.grpo=false
    agent.cache_hidden_state=false
    agent.cache_mode=false
    agent.vlm_type=internvl
    agent.dit_type=small
    agent.vlm_size=small
    agent.sampling_method=ddim
    "agent.metric_cache_path=${METRIC_CACHE}"
  )
  if [[ -n "${MAX_SCENES}" ]]; then
    agent_overrides+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
  fi

  echo "[eval] bucket=${bucket} split=${split}"
  echo "[eval] checkpoint=${checkpoint}"
  if [[ "${METRIC}" == "pdms" ]]; then
    scene_filter="${NAVSIM_V2_ROOT}/navsim/planning/script/config/common/train_test_split/scene_filter/navtest_${bucket}.yaml"
    need_file "${scene_filter}"
    NAVSIM_DEVKIT_ROOT="${REPO_ROOT}" \
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" \
      "${RUNNER}" "${agent_overrides[@]}" \
      "+scene_filter_path=${scene_filter}" \
      "experiment_name=${EXPERIMENT_PREFIX}-${bucket}" \
      "metric_cache_path=${METRIC_CACHE}" worker=sequential
  else
    v2_overrides=(
      "${agent_overrides[@]}"
      "agent.vlm_weights_path=${VLM_WEIGHTS_PATH}"
      "train_test_split=${split}"
      "experiment_name=${EXPERIMENT_PREFIX}-${bucket}"
      "metric_cache_path=${METRIC_CACHE}"
      worker=sequential
    )
    if [[ "${METRIC}" == "navhard" ]]; then
      v2_overrides+=(
        "synthetic_sensor_path=${NAVHARD_SENSOR_PATH}"
        "synthetic_scenes_path=${NAVHARD_SCENES_PATH}"
      )
    fi
    NAVSIM_DEVKIT_ROOT="${NAVSIM_V2_ROOT}" \
    PYTHONPATH="${NAVSIM_V2_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" \
      "${RUNNER}" "${v2_overrides[@]}"
  fi
done

#!/usr/bin/env bash
# GoalBridge teacher-guidance probe: is the routed teacher's x0 sane at the
# states the student actually visits?
#
# Pure inference on ONE GPU. Trains nothing, writes no checkpoint.
#
# Runs two checkpoints so the gap can be read as "already broken at init" vs
# "degraded during OPD":
#
#   opd_after : the GoalBridge student you trained. student_goal_source=pred.
#   base_il   : the goal-free IL checkpoint OPD starts from.
#               student_goal_source=none -- its goal head and goal encoder were
#               never trained, and for goal_mode=adaln the goal encoder's last
#               layer is zero-initialised, so conditioning on its "prediction"
#               is identical to goal-free anyway. The pred-goal (dark yellow)
#               condition is therefore skipped for this run.
#
# Outputs per run, under ${OUT_DIR}:
#   <tag>_payload.pt   raw probe tensors (replot without re-running inference)
#   <tag>_summary.png  per-step ADE / clamp saturation / z_t second-moment ratio
#   <tag>_bev_by_error/{q1_low,...,q4_high}/*.png
#                       one sample per image, with z1..z4 shown left-to-right
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/output/tensorboard}"
export PATH="${CONDA_BIN:-/opt/conda/envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${OUTPUT_BASE}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export SCENE_ROUTER_BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${REPO_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/recdrive/bin/python}"

# The GoalBridge student under test. Point this at the checkpoint whose viz
# looked wrong; it is the run being explained.
BASE_IL_CKPT="${BASE_IL_CKPT:-/mnt/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
STUDENT_CKPT="${STUDENT_CKPT:-/mnt/datasets/recdrive/20260513/exp2/exp/training_goalbridge_goal_kd_ungated_opd_v1/2026.09.08.03.52.22/checkpoints/epoch=47-step=136128.ckpt}"

TEACHER_GOAL_MODE="${TEACHER_GOAL_MODE:-adaln}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/mnt/datasets/recdrive/20260513/exp2/exp/training_teacher_general_or_no_tag_200il_30goal_adaln_newvlm/2026.09.08.07.58.19/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/mnt/datasets/recdrive/20260513/exp2/exp/training_teacher_progress_curbside_stopgo_200il_30goal_adaln_newvlm/2026.09.08.07.52.51/lightning_logs/version_0/checkpoints/epoch=197-step=91872.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/mnt/datasets/recdrive/20260513/exp2/exp/training_teacher_rule_intersection_200il_30goal_adaln_newvlm/2026.09.08.07.52.51/lightning_logs/version_0/checkpoints/epoch=198-step=62486.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/mnt/datasets/recdrive/20260513/exp2/exp/training_teacher_safety_dynamics_interaction_200il_30goal_adaln_newvlm/2026.09.08.07.58.36/lightning_logs/version_0/checkpoints/epoch=198-step=38606.ckpt}"

VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/mnt/datasets/simscale/20260709/data/simscale}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"
NAV_TOKEN_JSON="${NAV_TOKEN_JSON:-/mnt/datasets/simscale/20260709/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"
SIM0_TOKEN_JSON="${SIM0_TOKEN_JSON:-${SIMSCALE_BUCKET_ROOT}/scene_buckets_synthetic_reaction_pdm_v1.0-0_quality/exclusive_token_to_bucket.json}"
SIM1_TOKEN_JSON="${SIM1_TOKEN_JSON:-${SIMSCALE_BUCKET_ROOT}/scene_buckets_synthetic_reaction_pdm_v1.0-1_quality/exclusive_token_to_bucket.json}"
DIRECT_INDEX_ROOT="${DIRECT_INDEX_ROOT:-${SIMSCALE_BUCKET_ROOT}/il_training_newvlm_direct}"
SAFETY_TRAIN_INDEX="${SAFETY_TRAIN_INDEX:-${DIRECT_INDEX_ROOT}/safety_dynamics_interaction/train_index.json}"
RULE_TRAIN_INDEX="${RULE_TRAIN_INDEX:-${DIRECT_INDEX_ROOT}/rule_intersection/train_index.json}"
PROGRESS_TRAIN_INDEX="${PROGRESS_TRAIN_INDEX:-${DIRECT_INDEX_ROOT}/progress_curbside_stopgo/train_index.json}"
GENERAL_TRAIN_INDEX="${GENERAL_TRAIN_INDEX:-${DIRECT_INDEX_ROOT}/general_or_no_tag/train_index.json}"

AGENT="${AGENT:-recogdrive_agent_goalbridge_pred_goal_kd_opd}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# Balanced diagnostic set: four routed buckets x 64 samples = 256 per run.
SAMPLES_PER_BUCKET="${SAMPLES_PER_BUCKET:-64}"
SEED="${SEED:-0}"
# Which runs to do: "both" (default), "after", or "base".
RUNS="${RUNS:-both}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-goalbridge_guidance_probe}"
OUT_DIR="${OUT_DIR:-${OUTPUT_BASE}/${EXPERIMENT_NAME}}"
mkdir -p "${OUT_DIR}"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] required file missing: $1" >&2; exit 1; }; }
require_dir()  { [[ -d "$1" ]] || { echo "[ERROR] required directory missing: $1" >&2; exit 1; }; }

for path in \
  "${BASE_IL_CKPT}" "${STUDENT_CKPT}" \
  "${TEACHER_GENERAL_CKPT}" "${TEACHER_PROGRESS_CKPT}" \
  "${TEACHER_RULE_CKPT}" "${TEACHER_SAFETY_CKPT}" \
  "${NAV_TOKEN_JSON}" "${SIM0_TOKEN_JSON}" "${SIM1_TOKEN_JSON}" \
  "${SAFETY_TRAIN_INDEX}" "${RULE_TRAIN_INDEX}" \
  "${PROGRESS_TRAIN_INDEX}" "${GENERAL_TRAIN_INDEX}"; do
  require_file "${path}"
done
require_dir "${VLM_PATH}"
require_dir "${NAV_CACHE_PATH}"

TOKEN_JSONS_ARG="[${NAV_TOKEN_JSON},${SIM0_TOKEN_JSON},${SIM1_TOKEN_JSON}]"
TRAIN_INDEXES_ARG="[${SAFETY_TRAIN_INDEX},${RULE_TRAIN_INDEX},${PROGRESS_TRAIN_INDEX},${GENERAL_TRAIN_INDEX}]"

run_probe() {
  local tag="$1" ckpt="$2" goal_source="$3"
  echo "======================================================================"
  echo "[guidance] tag=${tag} goal_source=${goal_source}"
  echo "[guidance] ckpt=${ckpt}"
  echo "======================================================================"
  "${PYTHON_BIN}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_goalbridge_guidance_diagnostic.py" \
    agent="${AGENT}" \
    "agent.checkpoint_path='${ckpt}'" \
    "agent.vlm_path='${VLM_PATH}'" \
    "agent.teacher_goal_mode='${TEACHER_GOAL_MODE}'" \
    "agent.teacher_ckpt_progress_curbside_stopgo='${TEACHER_PROGRESS_CKPT}'" \
    "agent.teacher_ckpt_rule_intersection='${TEACHER_RULE_CKPT}'" \
    "agent.teacher_ckpt_safety_dynamics_interaction='${TEACHER_SAFETY_CKPT}'" \
    "agent.teacher_ckpt_general_or_no_tag='${TEACHER_GENERAL_CKPT}'" \
    "agent.token_to_bucket_json=${TOKEN_JSONS_ARG}" \
    agent.collect_viz=false \
    "+scene_router_direct_train_indexes=${TRAIN_INDEXES_ARG}" \
    "+guidance.out_dir='${OUT_DIR}'" \
    "+guidance.tag='${tag}'" \
    "+guidance.student_goal_source='${goal_source}'" \
    "+guidance.samples_per_bucket=${SAMPLES_PER_BUCKET}" \
    "+guidance.seed=${SEED}" \
    dataloader.params.batch_size="${BATCH_SIZE}" \
    experiment_name="${EXPERIMENT_NAME}_${tag}" \
    train_test_split="${TRAIN_TEST_SPLIT:-navtrain}" \
    cache_path="${NAV_CACHE_PATH}" \
    use_cache_without_dataset=true \
    force_cache_computation=false \
    hydra/job_logging=stdout \
    hydra.output_subdir=null 2>&1 | tee "${OUT_DIR}/${tag}.log"
}

if [[ "${RUNS}" == "both" || "${RUNS}" == "after" ]]; then
  run_probe "opd_after" "${STUDENT_CKPT}" "pred"
fi
if [[ "${RUNS}" == "both" || "${RUNS}" == "base" ]]; then
  run_probe "base_il" "${BASE_IL_CKPT}" "none"
fi

echo
echo "[guidance] outputs in ${OUT_DIR}"
echo "[guidance] replot without re-running inference:"
echo "  ${PYTHON_BIN} -m navsim.planning.training.goalbridge_guidance_plot \\"
echo "      ${OUT_DIR}/opd_after_payload.pt"

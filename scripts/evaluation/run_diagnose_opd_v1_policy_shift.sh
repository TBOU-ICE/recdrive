#!/usr/bin/env bash
# Measure whether v1 Goal-OPD recovered the teacher's privileged policy shift.
#
#   P_base = new-VLM IL init (the student before OPD)
#   P_T    = routed goal-conditioned IL teachers (adaln, epoch=199)
#   P_S    = OPD student (default: v4_resume22 last)
#
# Uses the v1-gpt planner code that produced those checkpoints. This repo only
# hosts the diagnostic; recdrive-multi-opd-v1-gpt is not modified.
#
#   N_PER_BUCKET=100 bash scripts/evaluation/run_diagnose_opd_v1_policy_shift.sh
#   STUDENT_CKPT=.../epoch=22-step=65228.ckpt EXPERIMENT_NAME=..._v4e22 \
#     bash scripts/evaluation/run_diagnose_opd_v1_policy_shift.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
V1_ROOT="${V1_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1-gpt}"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${V1_ROOT}}"
export PATH="${CONDA_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin}:$PATH"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
MANIFEST="${MANIFEST:-${V1_ROOT}/data/epdms/manifests/nav_train_newvlm.json}"
TOKEN_TO_BUCKET_JSON="${TOKEN_TO_BUCKET_JSON:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"
BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${V1_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"

BASE_CKPT="${BASE_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
STUDENT_CKPT="${STUDENT_CKPT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_scene_router_dit_goal_opd_v4_resume22/2026.08.26.06.03.01/lightning_logs/version_0/checkpoints/last.ckpt}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_progress_curbside_stopgo_il_goal_adaln_newvlm/2026.08.04.10.54.30/lightning_logs/version_0/checkpoints/epoch=199-step=92800.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_rule_intersection_il_goal_adaln_newvlm/2026.08.02.16.39.57/lightning_logs/version_0/checkpoints/epoch=199-step=62800.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_safety_dynamics_interaction_il_goal_adaln_newvlm/2026.08.04.10.25.26/lightning_logs/version_0/checkpoints/epoch=199-step=38800.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_general_or_no_tag_il_goal_adaln_newvlm/2026.08.04.10.44.01/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"

N_PER_BUCKET="${N_PER_BUCKET:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-diagnose_opd_v1_policy_shift_resume22}"
OUT_DIR="${OUT_DIR:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/run.log}"
mkdir -p "${OUT_DIR}"

for f in \
  "${NAVSIM_DEVKIT_ROOT}" \
  "${CACHE_PATH}" \
  "${MANIFEST}" \
  "${TOKEN_TO_BUCKET_JSON}" \
  "${BASE_CKPT}" \
  "${STUDENT_CKPT}" \
  "${TEACHER_PROGRESS_CKPT}" \
  "${TEACHER_RULE_CKPT}" \
  "${TEACHER_SAFETY_CKPT}" \
  "${TEACHER_GENERAL_CKPT}" \
  "${SCRIPT_DIR}/diagnose_opd_policy_shift.py"
do
  [[ -e "${f}" ]] || { echo "[policy-shift] missing: ${f}" >&2; exit 2; }
done

echo "[policy-shift] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "[policy-shift] STUDENT_CKPT=${STUDENT_CKPT}"
echo "[policy-shift] N_PER_BUCKET=${N_PER_BUCKET} OUT_DIR=${OUT_DIR}"

python "${SCRIPT_DIR}/diagnose_opd_policy_shift.py" \
  --cache-path "${CACHE_PATH}" \
  --manifest "${MANIFEST}" \
  --token-to-bucket-json "${TOKEN_TO_BUCKET_JSON}" \
  --base-ckpt "${BASE_CKPT}" \
  --student-ckpt "${STUDENT_CKPT}" \
  --teacher-progress-ckpt "${TEACHER_PROGRESS_CKPT}" \
  --teacher-rule-ckpt "${TEACHER_RULE_CKPT}" \
  --teacher-safety-ckpt "${TEACHER_SAFETY_CKPT}" \
  --teacher-general-ckpt "${TEACHER_GENERAL_CKPT}" \
  --teacher-goal-mode "${TEACHER_GOAL_MODE:-adaln}" \
  --student-adaln-bound "${STUDENT_ADALN_BOUND:-8.0}" \
  --n-per-bucket "${N_PER_BUCKET}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --bad-cache-list "${BAD_CACHE_LIST}" \
  --out-dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

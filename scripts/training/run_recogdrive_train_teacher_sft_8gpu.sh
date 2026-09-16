#!/usr/bin/env bash
# Phase B -- teacher-rollout SFT for the deployable GoalBridge student.
#
# Target   : the teacher trajectory cached in Phase A (never the GT trajectory).
# Condition: that trajectory's OWN endpoint, so the goal and the target stay
#            mutually consistent and goal_encoder finally receives gradient.
# Init     : the same goal-free fullmix IL checkpoint OPD uses today.
# Output   : a checkpoint that drops straight into Phase C as agent.checkpoint_path.
#
# Data is read by absolute path from the original /mnt caches via the prebuilt
# direct indexes -- no symlinks, no copies.  Everything written lands under
# OUTPUT_ROOT (/workspace/output/tensorboard).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export PATH="${CONDA_BIN:-/opt/conda/envs/recdrive/bin}:$PATH"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/output/tensorboard}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${OUTPUT_ROOT}}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export CACHE_READ_MAX_RETRIES="${CACHE_READ_MAX_RETRIES:-10}"
export RESILIENT_CACHE_LOADING="${RESILIENT_CACHE_LOADING:-1}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23612}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_student_to_teacher_sft_goalbridge_v1}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}/sft.log}"
mkdir -p "$(dirname "${LOG_FILE}")" "${TENSORBOARD_DIR}"

# ---- Phase A output -------------------------------------------------------
ROLLOUT_EXPERIMENT="${ROLLOUT_EXPERIMENT:-teacher_rollout_v1}"
ROLLOUT_PATH="${ROLLOUT_PATH:-/mnt/datasets/recdrive/20260513/teacher_rollout_v1/teacher_rollout.pt}"
# The teacher is not an oracle; regressing onto its failures teaches the student
# nothing OPD would later reward. Set to 0 to keep every rollout.
MAX_TEACHER_FDE_M="${MAX_TEACHER_FDE_M:-3.0}"

# ---- data (direct absolute paths into the original caches) ----------------
INDEX_ROOT="${INDEX_ROOT:-/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm_direct}"
INDEX_NAME="${INDEX_NAME:-train_index.json}"
VAL_INDEX_PATH="${VAL_INDEX_PATH:-${INDEX_ROOT}/navtrain_full_val_index.json}"
VAL_CACHE_PATH="${VAL_CACHE_PATH:-/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"

# ---- model / optimisation -------------------------------------------------
BASE_CKPT="${BASE_CKPT:-/mnt/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"
VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"

GOAL_DROPOUT_P="${GOAL_DROPOUT_P:-0.1}"
GOAL_NOISE_P="${GOAL_NOISE_P:-0.2}"
GOAL_NOISE_STD_XY="${GOAL_NOISE_STD_XY:-1.0}"
GOAL_NOISE_STD_HEADING="${GOAL_NOISE_STD_HEADING:-0.1}"
GOAL_LOSS_WEIGHT="${GOAL_LOSS_WEIGHT:-1.0}"
TRAJ_EVAL_INTERVAL="${TRAJ_EVAL_INTERVAL:-50}"

LR="${LR:-1e-4}"
# Teacher rollouts are a static off-policy label set. SFT is a warm start, not
# the main course: training it to convergence overfits the student onto the
# teacher's fixed outputs and costs plasticity that OPD needs.
MAX_EPOCHS="${MAX_EPOCHS:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

[[ -f "${ROLLOUT_PATH}" ]] || {
  echo "[ERROR] teacher rollout cache missing: ${ROLLOUT_PATH}" >&2
  echo "        Run scripts/generation/run_teacher_rollout_cache_8gpu.sh first." >&2
  exit 1; }
[[ -f "${VAL_INDEX_PATH}" ]] || { echo "[ERROR] val index missing: ${VAL_INDEX_PATH}" >&2; exit 1; }
[[ -d "${VAL_CACHE_PATH}" ]] || { echo "[ERROR] val cache missing: ${VAL_CACHE_PATH}" >&2; exit 1; }
for b in general_or_no_tag progress_curbside_stopgo rule_intersection safety_dynamics_interaction; do
  [[ -f "${INDEX_ROOT}/${b}/${INDEX_NAME}" ]] || {
    echo "[ERROR] direct index missing: ${INDEX_ROOT}/${b}/${INDEX_NAME}" >&2; exit 1; }
done

if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[teacher-sft] RESUME full trainer state from: ${RESUME_CKPT}"
  CKPT_ARGS=( "agent.checkpoint_path=null" "+ckpt_path='${RESUME_CKPT}'" )
else
  echo "[teacher-sft] FRESH warm-start from BASE_CKPT: ${BASE_CKPT}"
  [[ -f "${BASE_CKPT}" ]] || { echo "[ERROR] BASE_CKPT missing: ${BASE_CKPT}" >&2; exit 1; }
  CKPT_ARGS=( "agent.checkpoint_path='${BASE_CKPT}'" )
fi

echo "======================================================================"
echo "[teacher-sft] target=cached teacher rollout, condition=teacher endpoint (no GT goal)"
echo "[teacher-sft] ROLLOUT_PATH=${ROLLOUT_PATH}  MAX_TEACHER_FDE_M=${MAX_TEACHER_FDE_M}"
echo "[teacher-sft] INDEX_ROOT=${INDEX_ROOT}  (direct absolute paths, no links)"
echo "[teacher-sft] goal corruption: masked=${GOAL_DROPOUT_P} noisy=${GOAL_NOISE_P} noise_xy=${GOAL_NOISE_STD_XY}m"
echo "[teacher-sft] LR=${LR}  MAX_EPOCHS=${MAX_EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  GPUS=${GPUS}"
echo "[teacher-sft] NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}  TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "[teacher-sft] watch train/student_fde_teacher_m (should fall) and"
echo "[teacher-sft]       train/goal_branch_weight_norm (must leave 0.0)"
echo "======================================================================"

TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/recdrive/bin/torchrun}"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_teacher_sft.py" \
  agent=recogdrive_agent_teacher_sft \
  "${CKPT_ARGS[@]}" \
  "agent.vlm_path='${VLM_PATH}'" \
  agent.goal_dropout_p="${GOAL_DROPOUT_P}" \
  agent.goal_noise_p="${GOAL_NOISE_P}" \
  agent.goal_noise_std_xy="${GOAL_NOISE_STD_XY}" \
  agent.goal_noise_std_heading="${GOAL_NOISE_STD_HEADING}" \
  agent.goal_loss_weight="${GOAL_LOSS_WEIGHT}" \
  agent.traj_eval_interval="${TRAJ_EVAL_INTERVAL}" \
  agent.lr="${LR}" \
  agent.sft_epochs="${MAX_EPOCHS}" \
  agent.sft_warmup_epochs="${WARMUP_EPOCHS}" \
  "+teacher_sft.index_root='${INDEX_ROOT}'" \
  "+teacher_sft.index_name='${INDEX_NAME}'" \
  "+teacher_sft.rollout_path='${ROLLOUT_PATH}'" \
  "+teacher_sft.max_teacher_fde_m=${MAX_TEACHER_FDE_M}" \
  "+teacher_sft.val_index_path='${VAL_INDEX_PATH}'" \
  "+teacher_sft.val_cache_path='${VAL_CACHE_PATH}'" \
  "+tensorboard_dir='${TENSORBOARD_DIR}'" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  +dataloader.params.persistent_workers=true \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  use_cache_without_dataset=true \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

echo "[teacher-sft] done. Feed the best checkpoint to Phase C via STUDENT_CKPT."

#!/usr/bin/env bash
# Phase C: compare GoalBridge OPD from the SFT checkpoint against the same OPD
# from the original goal-free student.  Run INIT_MODE=sft and INIT_MODE=base as
# two independent 8-GPU jobs (normally on two machines).
#
# Objective: goal_loss + ungated reverse-KL only.
# Data: four direct indexes, no teacher-rollout/quality/bad-sample filtering.
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
# Retry the SAME cache item on transient I/O/decode failures.  This does not
# resample another token and therefore cannot change the audited sample order.
export CACHE_READ_MAX_RETRIES="${CACHE_READ_MAX_RETRIES:-10}"

NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23631}"

INIT_MODE="${INIT_MODE:-sft}"
SFT_CKPT="${SFT_CKPT:-/mnt/datasets/recdrive/20260513/training_student_to_teacher_sft_goalbridge_v1/2026.09.16.07.59.36/checkpoints/epoch=9-step=14190.ckpt}"
BASE_CKPT="${BASE_CKPT:-/mnt/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"
case "${INIT_MODE}" in
  sft)
    STUDENT_CKPT="${STUDENT_CKPT:-${SFT_CKPT}}"
    DEFAULT_EXPERIMENT="training_student_to_teacher_sft_opd_sft_init_v1"
    ;;
  base|nosft)
    INIT_MODE="base"
    STUDENT_CKPT="${STUDENT_CKPT:-${BASE_CKPT}}"
    DEFAULT_EXPERIMENT="training_student_to_teacher_sft_opd_base_init_v1"
    ;;
  *)
    echo "[ERROR] INIT_MODE must be sft or base, got: ${INIT_MODE}" >&2
    exit 2
    ;;
esac

TEACHER_GOAL_MODE="${TEACHER_GOAL_MODE:-adaln}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_progress_curbside_stopgo_il_goal_adaln_newvlm/2026.08.04.10.54.30/lightning_logs/version_0/checkpoints/epoch=199-step=92800.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_rule_intersection_il_goal_adaln_newvlm/2026.08.02.16.39.57/lightning_logs/version_0/checkpoints/epoch=199-step=62800.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_safety_dynamics_interaction_il_goal_adaln_newvlm/2026.08.04.10.25.26/lightning_logs/version_0/checkpoints/epoch=199-step=38800.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_general_or_no_tag_il_goal_adaln_newvlm/2026.08.04.10.44.01/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"

VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
INDEX_ROOT="${INDEX_ROOT:-/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm_direct}"
INDEX_NAME="${INDEX_NAME:-train_index.json}"
VAL_INDEX_PATH="${VAL_INDEX_PATH:-${INDEX_ROOT}/navtrain_full_val_index.json}"
VAL_CACHE_PATH="${VAL_CACHE_PATH:-/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
TOKEN_TO_BUCKET_JSON="${TOKEN_TO_BUCKET_JSON:-/mnt/datasets/simscale/20260709/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"
PROBE_INDEX_PATH="${PROBE_INDEX_PATH:-${REPO_ROOT}/data/phase_c/student_to_teacher_opd_probe_4x32_seed0.json}"

GOAL_LOSS_WEIGHT="${GOAL_LOSS_WEIGHT:-1.0}"
KD_WEIGHT="${KD_WEIGHT:-1.0}"
KL_PRECISION_CLIP="${KL_PRECISION_CLIP:-25.0}"
STUDENT_ADALN_BOUND="${STUDENT_ADALN_BOUND:-0.0}"
LR="${LR:-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4}"
PROBE_EVERY_N_EPOCHS="${PROBE_EVERY_N_EPOCHS:-1}"
PROBE_INTERVAL_STEPS="${PROBE_INTERVAL_STEPS:-0}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}/phase_c.log}"
mkdir -p "$(dirname "${LOG_FILE}")" "${TENSORBOARD_DIR}"

required_files=(
  "${STUDENT_CKPT}"
  "${TEACHER_PROGRESS_CKPT}"
  "${TEACHER_RULE_CKPT}"
  "${TEACHER_SAFETY_CKPT}"
  "${TEACHER_GENERAL_CKPT}"
  "${VAL_INDEX_PATH}"
  "${TOKEN_TO_BUCKET_JSON}"
  "${PROBE_INDEX_PATH}"
)
for path in "${required_files[@]}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] required file missing: ${path}" >&2; exit 1; }
done
for bucket in general_or_no_tag progress_curbside_stopgo rule_intersection safety_dynamics_interaction; do
  [[ -f "${INDEX_ROOT}/${bucket}/${INDEX_NAME}" ]] || {
    echo "[ERROR] direct index missing: ${INDEX_ROOT}/${bucket}/${INDEX_NAME}" >&2
    exit 1
  }
done

if [[ -n "${RESUME_CKPT}" ]]; then
  [[ -f "${RESUME_CKPT}" ]] || { echo "[ERROR] resume checkpoint missing: ${RESUME_CKPT}" >&2; exit 1; }
  CKPT_ARGS=("agent.checkpoint_path=null" "+ckpt_path='${RESUME_CKPT}'")
else
  CKPT_ARGS=("agent.checkpoint_path='${STUDENT_CKPT}'")
fi

echo "======================================================================"
echo "[Phase-C] init=${INIT_MODE} student=${STUDENT_CKPT}"
echo "[Phase-C] objective=${GOAL_LOSS_WEIGHT}*goal + ${KD_WEIGHT}*reverse_KL"
echo "[Phase-C] anchor=off smooth=off recoverability_gate=off adaLN_bound=${STUDENT_ADALN_BOUND}"
echo "[Phase-C] data=full direct indexes; no rollout/quality/bad-sample filtering"
echo "[Phase-C] seed=${SEED} batch_per_gpu=${BATCH_SIZE} GPUs=${GPUS} epochs=${MAX_EPOCHS}"
echo "[Phase-C] fixed probe=4x32; detailed viz=4x3; probe every ${PROBE_EVERY_N_EPOCHS} epoch(s)"
echo "[Phase-C] compare run_manifest.json and epoch_order_sha256.jsonl across both runs"
echo "[Phase-C] source SHA256:"
sha256sum "${PROBE_INDEX_PATH}" "${VAL_INDEX_PATH}"
for bucket in general_or_no_tag progress_curbside_stopgo rule_intersection safety_dynamics_interaction; do
  sha256sum "${INDEX_ROOT}/${bucket}/${INDEX_NAME}"
done
echo "======================================================================"

TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/recdrive/bin/torchrun}"
"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_student_to_teacher_sft_opd.py" \
  agent=recogdrive_agent_goalbridge_goal_kd_opd \
  "${CKPT_ARGS[@]}" \
  "agent.vlm_path='${VLM_PATH}'" \
  "agent.teacher_goal_mode='${TEACHER_GOAL_MODE}'" \
  "agent.teacher_ckpt_progress_curbside_stopgo='${TEACHER_PROGRESS_CKPT}'" \
  "agent.teacher_ckpt_rule_intersection='${TEACHER_RULE_CKPT}'" \
  "agent.teacher_ckpt_safety_dynamics_interaction='${TEACHER_SAFETY_CKPT}'" \
  "agent.teacher_ckpt_general_or_no_tag='${TEACHER_GENERAL_CKPT}'" \
  "agent.token_to_bucket_json='${TOKEN_TO_BUCKET_JSON}'" \
  agent.goal_loss_weight="${GOAL_LOSS_WEIGHT}" \
  agent.kd_weight="${KD_WEIGHT}" \
  agent.kl_precision_clip="${KL_PRECISION_CLIP}" \
  agent.student_adaln_bound="${STUDENT_ADALN_BOUND}" \
  agent.lr="${LR}" \
  agent.collect_viz=false \
  agent.viz_interval_steps=0 \
  "+phase_c.repo_root='${REPO_ROOT}'" \
  "+phase_c.index_root='${INDEX_ROOT}'" \
  "+phase_c.index_name='${INDEX_NAME}'" \
  "+phase_c.val_index_path='${VAL_INDEX_PATH}'" \
  "+phase_c.val_cache_path='${VAL_CACHE_PATH}'" \
  "+phase_c.probe_index_path='${PROBE_INDEX_PATH}'" \
  "+phase_c.init_mode='${INIT_MODE}'" \
  "+phase_c.student_checkpoint='${STUDENT_CKPT}'" \
  "+phase_c.probe_batch_size=${PROBE_BATCH_SIZE}" \
  +phase_c.probe_per_bucket=32 \
  +phase_c.probe_viz_per_bucket=3 \
  "+phase_c.probe_every_n_epochs=${PROBE_EVERY_N_EPOCHS}" \
  "+phase_c.probe_interval_steps=${PROBE_INTERVAL_STEPS}" \
  "+tensorboard_dir='${TENSORBOARD_DIR}'" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  dataloader.params.prefetch_factor=2 \
  +dataloader.params.persistent_workers=true \
  seed="${SEED}" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  hydra/job_logging=stdout \
  hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

echo "[Phase-C] done: ${EXPERIMENT_NAME}"

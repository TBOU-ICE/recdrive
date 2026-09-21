#!/usr/bin/env bash
# Goal-free scene-router OPD: reverse-KL KD is the only training objective.
# All implementation/config files are additive; the original scene-router job
# remains unchanged.
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
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export SCENE_ROUTER_BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${REPO_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"

NNODES="${NNODES:-${WORLD_SIZE:-1}}"
RANK="${RANK:-0}"
GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23517}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/recdrive/bin/torchrun}"

# Goal-free new-VLM IL student and the original goal-free routed RL teachers.
STUDENT_CKPT="${STUDENT_CKPT:-/m n trecdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_progress_curbside_stopgo_rl_newvlm_v1-1/2026.07.30.11.52.25/lightning_logs/version_0/checkpoints/epoch=6-step=1113.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_rule_intersection_rl_newvlm_v1-1/2026.07.30.11.52.22/lightning_logs/version_0/checkpoints/epoch=12-step=1833.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_safety_dynamics_interaction_rl_newvlm_v1-1/2026.07.30.11.52.26/lightning_logs/version_0/checkpoints/epoch=13-step=1764.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/mnt/models/recdrive/v1.0.0/training_teacher_general_or_no_tag_rl_newvlm_v1-1/2026.07.30.11.52.29/lightning_logs/version_0/checkpoints/epoch=4-step=875.ckpt}"

# Direct-index navtrain + SimScale 0/1 data; no symlink or manifest is needed.
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
VAL_INDEX_PATH="${VAL_INDEX_PATH:-${DIRECT_INDEX_ROOT}/navtrain_full_val_index.json}"

KD_WEIGHT="${KD_WEIGHT:-1.0}"
KL_PRECISION_CLIP="${KL_PRECISION_CLIP:-25.0}"
COLLECT_VIZ="${COLLECT_VIZ:-1}"
VIZ_INTERVAL_STEPS="${VIZ_INTERVAL_STEPS:-1000}"
LR="${LR:-1e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] required file missing: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] required directory missing: $1" >&2; exit 1; }; }
for path in \
  "${STUDENT_CKPT}" "${TEACHER_PROGRESS_CKPT}" "${TEACHER_RULE_CKPT}" \
  "${TEACHER_SAFETY_CKPT}" "${TEACHER_GENERAL_CKPT}" "${NAV_TOKEN_JSON}" \
  "${SIM0_TOKEN_JSON}" "${SIM1_TOKEN_JSON}" "${SAFETY_TRAIN_INDEX}" \
  "${RULE_TRAIN_INDEX}" "${PROGRESS_TRAIN_INDEX}" "${GENERAL_TRAIN_INDEX}" \
  "${VAL_INDEX_PATH}"; do
  require_file "${path}"
done
[[ -z "${RESUME_CKPT}" ]] || require_file "${RESUME_CKPT}"
for path in "${VLM_PATH}" "${NAV_CACHE_PATH}"; do
  require_dir "${path}"
done

TOKEN_JSONS_ARG="[${NAV_TOKEN_JSON},${SIM0_TOKEN_JSON},${SIM1_TOKEN_JSON}]"
TRAIN_INDEXES_ARG="[${SAFETY_TRAIN_INDEX},${RULE_TRAIN_INDEX},${PROGRESS_TRAIN_INDEX},${GENERAL_TRAIN_INDEX}]"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_il_rl_teacher_kd_only_reverse_kl_v1}"
OUTPUT_ROOT="${OUTPUT_BASE}/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/run.log}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_BASE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints}"
mkdir -p "$(dirname "${LOG_FILE}")" "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

HYDRA_RESUME=()
[[ -z "${RESUME_CKPT}" ]] || HYDRA_RESUME+=("+ckpt_path='${RESUME_CKPT}'")

echo "[SceneRouter-KDOnly] student=${STUDENT_CKPT}"
echo "[SceneRouter-KDOnly] goal=none exopd=off smooth=off objective=${KD_WEIGHT}*reverse_kl"
echo "[SceneRouter-KDOnly] epochs=${MAX_EPOCHS} scheduler=${SCHEDULER_EPOCHS} warmup=${WARMUP_EPOCHS} lr=${LR}"
echo "[SceneRouter-KDOnly] viz=${COLLECT_VIZ}/${VIZ_INTERVAL_STEPS}steps"
echo "[SceneRouter-KDOnly] output=${OUTPUT_ROOT}"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_scene_router_kd_only.py" \
  agent=recogdrive_agent_scene_router_kd_only \
  "agent.checkpoint_path='${STUDENT_CKPT}'" \
  "agent.vlm_path='${VLM_PATH}'" \
  "agent.teacher_ckpt_progress_curbside_stopgo='${TEACHER_PROGRESS_CKPT}'" \
  "agent.teacher_ckpt_rule_intersection='${TEACHER_RULE_CKPT}'" \
  "agent.teacher_ckpt_safety_dynamics_interaction='${TEACHER_SAFETY_CKPT}'" \
  "agent.teacher_ckpt_general_or_no_tag='${TEACHER_GENERAL_CKPT}'" \
  "agent.token_to_bucket_json=${TOKEN_JSONS_ARG}" \
  "+scene_router_direct_train_indexes=${TRAIN_INDEXES_ARG}" \
  "+scene_router_direct_val_index='${VAL_INDEX_PATH}'" \
  agent.exopd_ref_checkpoint='' \
  agent.exopd_lambda=1.0 \
  agent.scene_router_smooth_weight=0.0 \
  agent.kd_weight="${KD_WEIGHT}" \
  agent.kl_precision_clip="${KL_PRECISION_CLIP}" \
  agent.collect_viz="${COLLECT_VIZ}" \
  agent.viz_interval_steps="${VIZ_INTERVAL_STEPS}" \
  agent.lr="${LR}" \
  agent.scheduler_epochs="${SCHEDULER_EPOCHS}" \
  agent.warmup_epochs="${WARMUP_EPOCHS}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  dataloader.params.prefetch_factor=4 \
  +dataloader.params.persistent_workers=true \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  "+tensorboard_dir='${TENSORBOARD_DIR}'" \
  "+checkpoint_dir='${CHECKPOINT_DIR}'" \
  cache_path="${NAV_CACHE_PATH}" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  ${HYDRA_RESUME[@]+"${HYDRA_RESUME[@]}"} \
  hydra/job_logging=stdout \
  hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

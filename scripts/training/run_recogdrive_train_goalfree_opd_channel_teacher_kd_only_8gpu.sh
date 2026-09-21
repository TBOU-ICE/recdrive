#!/usr/bin/env bash
# Channel-teacher KD-only OPD ablation with a strictly goal-free student.
#   student: plain ReCogDriveDiffusionPlanner; no goal predictor/injection
#   teacher: routed privileged channel-injected expert conditioned on GT endpoint
#   loss:    full-strength shared-student-variance reverse-KL only
#   data:    four direct bucket indexes (navtrain + SimScale 0/1)
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
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/recdrive/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/recdrive/bin/python}"

# A caller-supplied port is mandatory for multi-node coordination. On one node,
# choose a free port to avoid colliding with another eight-GPU job.
if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ "${NNODES}" -gt 1 ]]; then
    MASTER_PORT=23513
  else
    MASTER_PORT="$("${PYTHON_BIN}" -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
  fi
fi

# Same deployable goal-free IL initialization as the channel baseline.
STUDENT_CKPT="${STUDENT_CKPT:-/mnt/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
STUDENT_ADALN_BOUND="${STUDENT_ADALN_BOUND:-8.0}"
RESUME_CKPT="${RESUME_CKPT:-}"

TEACHER_GOAL_MODE="${TEACHER_GOAL_MODE:-channel}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/mnt/datasets/recdrive/20260513/training_teacher_general_or_no_tag_200il_30goal_channel_newvlm/2026.09.14.07.06.41/lightning_logs/version_0/checkpoints/epoch=198-step=117410.ckpt}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/mnt/datasets/recdrive/20260513/training_teacher_progress_curbside_stopgo_200il_30goal_channel_newvlm/2026.09.14.07.01.33/lightning_logs/version_0/checkpoints/epoch=192-step=89552.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/mnt/datasets/recdrive/20260513/training_teacher_rule_intersection_200il_30goal_channel_newvlm/2026.09.14.07.01.29/lightning_logs/version_0/checkpoints/epoch=199-step=62800.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/mnt/datasets/recdrive/20260513/training_teacher_safety_dynamics_interaction_200il_30goal_channel_newvlm/2026.09.14.07.06.50/lightning_logs/version_0/checkpoints/epoch=199-step=38800.ckpt}"

# Same direct-index data contract as the channel baseline.
VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/mnt/datasets/simscale/20260709/data/simscale}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"
NAV_TOKEN_JSON="${NAV_TOKEN_JSON:-/mnt/datasets/simscale/20260709/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"
SIM0_CACHE="${SIM0_CACHE:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0}"
SIM1_CACHE="${SIM1_CACHE:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-1}"
SIM0_TOKEN_JSON="${SIM0_TOKEN_JSON:-${SIMSCALE_BUCKET_ROOT}/scene_buckets_synthetic_reaction_pdm_v1.0-0_quality/exclusive_token_to_bucket.json}"
SIM1_TOKEN_JSON="${SIM1_TOKEN_JSON:-${SIMSCALE_BUCKET_ROOT}/scene_buckets_synthetic_reaction_pdm_v1.0-1_quality/exclusive_token_to_bucket.json}"
DIRECT_INDEX_ROOT="${DIRECT_INDEX_ROOT:-${SIMSCALE_BUCKET_ROOT}/il_training_newvlm_direct}"
SAFETY_TRAIN_INDEX="${SAFETY_TRAIN_INDEX:-${DIRECT_INDEX_ROOT}/safety_dynamics_interaction/train_index.json}"
RULE_TRAIN_INDEX="${RULE_TRAIN_INDEX:-${DIRECT_INDEX_ROOT}/rule_intersection/train_index.json}"
PROGRESS_TRAIN_INDEX="${PROGRESS_TRAIN_INDEX:-${DIRECT_INDEX_ROOT}/progress_curbside_stopgo/train_index.json}"
GENERAL_TRAIN_INDEX="${GENERAL_TRAIN_INDEX:-${DIRECT_INDEX_ROOT}/general_or_no_tag/train_index.json}"
VAL_INDEX_PATH="${VAL_INDEX_PATH:-${DIRECT_INDEX_ROOT}/navtrain_full_val_index.json}"

KD_WEIGHT="${KD_WEIGHT:-20.0}"
KL_PRECISION_CLIP="${KL_PRECISION_CLIP:-25.0}"
COLLECT_VIZ="${COLLECT_VIZ:-1}"
VIZ_INTERVAL_STEPS="${VIZ_INTERVAL_STEPS:-1000}"
LR="${LR:-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-3}"
CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-5}"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] required file missing: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] required directory missing: $1" >&2; exit 1; }; }
for path in \
  "${STUDENT_CKPT}" "${TEACHER_GENERAL_CKPT}" "${TEACHER_PROGRESS_CKPT}" \
  "${TEACHER_RULE_CKPT}" "${TEACHER_SAFETY_CKPT}" "${NAV_TOKEN_JSON}" \
  "${SIM0_TOKEN_JSON}" "${SIM1_TOKEN_JSON}" "${SAFETY_TRAIN_INDEX}" \
  "${RULE_TRAIN_INDEX}" "${PROGRESS_TRAIN_INDEX}" "${GENERAL_TRAIN_INDEX}" \
  "${VAL_INDEX_PATH}"; do
  require_file "${path}"
done
[[ -z "${RESUME_CKPT}" ]] || require_file "${RESUME_CKPT}"
for path in "${VLM_PATH}" "${NAV_CACHE_PATH}" "${SIM0_CACHE}" "${SIM1_CACHE}"; do
  require_dir "${path}"
done

TOKEN_JSONS_ARG="[${NAV_TOKEN_JSON},${SIM0_TOKEN_JSON},${SIM1_TOKEN_JSON}]"
TRAIN_INDEXES_ARG="[${SAFETY_TRAIN_INDEX},${RULE_TRAIN_INDEX},${PROGRESS_TRAIN_INDEX},${GENERAL_TRAIN_INDEX}]"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_goalfree_opd_channel_teacher_kd_only_v1}"
OUTPUT_ROOT="${OUTPUT_BASE}/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/run.log}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_BASE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints}"
mkdir -p "$(dirname "${LOG_FILE}")" "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

HYDRA_RESUME=()
[[ -z "${RESUME_CKPT}" ]] || HYDRA_RESUME+=("+ckpt_path='${RESUME_CKPT}'")

echo "[GoalFree-KDOnly-Channel] student=${STUDENT_CKPT}"
echo "[GoalFree-KDOnly-Channel] student_goal=none teacher_goal_mode=${TEACHER_GOAL_MODE}"
echo "[GoalFree-KDOnly-Channel] student_adaln_bound=${STUDENT_ADALN_BOUND} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[GoalFree-KDOnly-Channel] objective=${KD_WEIGHT}*ungated_reverse_kl goal_predictor=absent"
echo "[GoalFree-KDOnly-Channel] viz=${COLLECT_VIZ}/${VIZ_INTERVAL_STEPS}steps output=${OUTPUT_ROOT}"
echo "[GoalFree-KDOnly-Channel] direct_index_root=${DIRECT_INDEX_ROOT}"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_scene_router_dit_goal_distill.py" \
  agent=recogdrive_agent_goalfree_kd_only_opd_channel \
  "agent.checkpoint_path='${STUDENT_CKPT}'" \
  "agent.vlm_path='${VLM_PATH}'" \
  "agent.teacher_goal_mode='${TEACHER_GOAL_MODE}'" \
  agent.student_adaln_bound="${STUDENT_ADALN_BOUND}" \
  "agent.teacher_ckpt_progress_curbside_stopgo='${TEACHER_PROGRESS_CKPT}'" \
  "agent.teacher_ckpt_rule_intersection='${TEACHER_RULE_CKPT}'" \
  "agent.teacher_ckpt_safety_dynamics_interaction='${TEACHER_SAFETY_CKPT}'" \
  "agent.teacher_ckpt_general_or_no_tag='${TEACHER_GENERAL_CKPT}'" \
  "agent.token_to_bucket_json=${TOKEN_JSONS_ARG}" \
  "+scene_router_direct_train_indexes=${TRAIN_INDEXES_ARG}" \
  "+scene_router_direct_val_index='${VAL_INDEX_PATH}'" \
  agent.kd_weight="${KD_WEIGHT}" \
  agent.kl_precision_clip="${KL_PRECISION_CLIP}" \
  agent.scene_router_smooth_weight=0.0 \
  agent.lr="${LR}" \
  agent.collect_viz="${COLLECT_VIZ}" \
  agent.viz_interval_steps="${VIZ_INTERVAL_STEPS}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  dataloader.params.prefetch_factor=4 \
  +dataloader.params.persistent_workers=true \
  +checkpoint_save_top_k="${CHECKPOINT_SAVE_TOP_K}" \
  +checkpoint_every_n_epochs="${CHECKPOINT_EVERY_N_EPOCHS}" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  "+tensorboard_dir='${TENSORBOARD_DIR}'" \
  "+checkpoint_dir='${CHECKPOINT_DIR}'" \
  "output_dir='${OUTPUT_ROOT}'" \
  cache_path="${NAV_CACHE_PATH}" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  ${HYDRA_RESUME[@]+"${HYDRA_RESUME[@]}"} \
  hydra/job_logging=stdout \
  hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

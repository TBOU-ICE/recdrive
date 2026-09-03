#!/usr/bin/env bash
# Privileged teacher pipeline: same strong base -> goal-aware mixed IL -> goal-aware bucket RL.
#
# One independent teacher is trained for each scene bucket. Select it with
# BUCKET_NAME (default: general_or_no_tag):
#   progress_curbside_stopgo | rule_intersection
#   safety_dynamics_interaction | general_or_no_tag
#
# IL example:
#   BUCKET_NAME=rule_intersection STAGE=il \
#     bash scripts/training/run_recogdrive_privileged_teacher_pipeline.sh
#
# RL example (old-script AdaLN IL checkpoint is the intended warm start):
#   BUCKET_NAME=rule_intersection STAGE=rl \
#   IL_CKPT=/path/to/old_goal_il.ckpt \
#     bash scripts/training/run_recogdrive_privileged_teacher_pipeline.sh
#
# Data: reuse the already-built no-goal mix on disk
#   il_training_newvlm/${BUCKET_NAME}_fullmix/metadata/train_index.json
# plus the other three buckets as the complement mix. No Alluxio walk, no new
# symlinks. Known-bad shards are dropped from the prebuilt list.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:${PATH}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export CACHE_READ_MAX_RETRIES="${CACHE_READ_MAX_RETRIES:-10}"
export CACHE_READ_RETRY_BASE_SEC="${CACHE_READ_RETRY_BASE_SEC:-0.5}"
export RESILIENT_CACHE_LOADING="${RESILIENT_CACHE_LOADING:-1}"
export CACHE_LOAD_MAX_RETRIES="${CACHE_LOAD_MAX_RETRIES:-8}"
export CACHE_LOAD_TIMEOUT_SEC="${CACHE_LOAD_TIMEOUT_SEC:-60}"
export BAD_CACHE_LIST="${BAD_CACHE_LIST:-${REPO_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"
export SCENE_ROUTER_BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${BAD_CACHE_LIST}}"

GPUS="${GPUS:-8}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23651}"
STAGE="${STAGE:-il}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

BUCKET_NAME="${BUCKET_NAME:-general_or_no_tag}"
case "${BUCKET_NAME}" in
  progress_curbside_stopgo|rule_intersection|safety_dynamics_interaction|general_or_no_tag) ;;
  *)
    echo "Unsupported BUCKET_NAME=${BUCKET_NAME}" >&2
    echo "Choose one of: progress_curbside_stopgo, rule_intersection, safety_dynamics_interaction, general_or_no_tag" >&2
    exit 2
    ;;
esac

NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain}"
BUCKET_TOKENS_JSON="${BUCKET_TOKENS_JSON:-${NAVTRAIN_OUTPUT_DIR}/exclusive_${BUCKET_NAME}_tokens.json}"
TOKEN_TO_LOG_JSON="${TOKEN_TO_LOG_JSON:-${NAVTRAIN_OUTPUT_DIR}/navtrain_token_to_buckets.json}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
BASE_CKPT="${BASE_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.20.14.31.06/lightning_logs/version_0/checkpoints/epoch=199-step=312200.ckpt}"
GOAL_MODE="${GOAL_MODE:-adaln}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/data/epdms/manifests}"
NAV_MANIFEST="${NAV_MANIFEST:-${MANIFEST_DIR}/nav_train_newvlm.json}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/simscale}"
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/workspace/datasets/simscale/20260709}"
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
SIMSCALE_METRIC_ROOT="${SIMSCALE_METRIC_ROOT:-/workspace/datasets/simscale/20260709}"
SIM_ROUNDS="${SIM_ROUNDS:-0,1}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_train_v2}"
MIX_ROOT_PARENT="${MIX_ROOT_PARENT:-/workspace/datasets/simscale/20260709/data/simscale/il_training_newvlm}"
MIX_ROOT="${MIX_ROOT:-${MIX_ROOT_PARENT}/${BUCKET_NAME}_fullmix}"
TRAIN_INDEX="${TRAIN_INDEX:-${MIX_ROOT}/metadata/train_index.json}"

join_hydra() {
  if [[ $# -eq 0 ]]; then
    echo "[]"
    return
  fi
  local out="["
  local first=1
  for item in "$@"; do
    if [[ "${first}" -eq 1 ]]; then
      first=0
    else
      out+=","
    fi
    out+="'${item}'"
  done
  out+="]"
  echo "${out}"
}

PEER_INDEX_LIST=()
for peer in progress_curbside_stopgo rule_intersection safety_dynamics_interaction general_or_no_tag; do
  if [[ "${peer}" == "${BUCKET_NAME}" ]]; then
    continue
  fi
  peer_index="${MIX_ROOT_PARENT}/${peer}_fullmix/metadata/train_index.json"
  if [[ -f "${peer_index}" ]]; then
    PEER_INDEX_LIST+=("${peer_index}")
  else
    echo "[privileged-teacher] ! missing peer mix index: ${peer_index}" >&2
  fi
done
PEER_INDEXES_ARG="$(join_hydra ${PEER_INDEX_LIST[@]+"${PEER_INDEX_LIST[@]}"})"

EXTRA_METRIC_LIST=()
IFS=',' read -r -a _sim_rounds <<< "${SIM_ROUNDS}"
for r in "${_sim_rounds[@]}"; do
  r="${r//[[:space:]]/}"
  [[ -z "${r}" ]] && continue
  metric_dir="${SIMSCALE_METRIC_ROOT}/metric_cache_synthetic_reaction_pdm_v1.0-${r}"
  if [[ -d "${metric_dir}" ]]; then
    EXTRA_METRIC_LIST+=("${metric_dir}")
  fi
done
EXTRA_METRICS_ARG="$(join_hydra ${EXTRA_METRIC_LIST[@]+"${EXTRA_METRIC_LIST[@]}"})"

IL_FULL_RATIO="${IL_FULL_RATIO:-0.5}"
IL_BUCKET_RATIO="${IL_BUCKET_RATIO:-0.5}"
RL_FULL_RATIO="${RL_FULL_RATIO:-0.5}"
RL_BUCKET_RATIO="${RL_BUCKET_RATIO:-0.5}"
IL_EPOCHS="${IL_EPOCHS:-8}"
RL_EPOCHS="${RL_EPOCHS:-10}"
IL_LR="${IL_LR:-5e-5}"
RL_LR="${RL_LR:-3e-5}"
BATCH_SIZE_IL="${BATCH_SIZE_IL:-16}"
BATCH_SIZE_RL="${BATCH_SIZE_RL:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

COMMON=(
  "agent=recogdrive_goal_agent"
  "agent.vlm_path='${VLM_PATH}'"
  "agent.cache_hidden_state=true"
  "agent.vlm_type='internvl'"
  "agent.dit_type='small'"
  "agent.vlm_size='small'"
  "agent.sampling_method='ddim'"
  "agent.goal_mode='${GOAL_MODE}'"
  "agent.goal_dropout_p=0.0"
  "bucket.name='${BUCKET_NAME}'"
  "bucket.tokens_json='${BUCKET_TOKENS_JSON}'"
  "bucket.token_to_log_json='${TOKEN_TO_LOG_JSON}'"
  "bucket.navtrain_output_dir='${NAVTRAIN_OUTPUT_DIR}'"
  "bucket.train_index='${TRAIN_INDEX}'"
  "bucket.peer_train_indexes=${PEER_INDEXES_ARG}"
  "bucket.extra_metric_cache_paths=${EXTRA_METRICS_ARG}"
  "train_test_split='${TRAIN_TEST_SPLIT}'"
  "cache_path='${CACHE_PATH}'"
  "use_cache_without_dataset=true"
  "force_cache_computation=false"
  "trainer.params.num_nodes=${NNODES}"
  "trainer.params.devices=${GPUS}"
  "trainer.params.strategy=ddp_find_unused_parameters_true"
  "trainer.params.precision=bf16-mixed"
  "dataloader.params.num_workers=${NUM_WORKERS}"
  "dataloader.params.prefetch_factor=${PREFETCH_FACTOR}"
  "+dataloader.params.persistent_workers=true"
)

run_torch() {
  /workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
    --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" --nproc_per_node="${GPUS}" "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory not found: $1" >&2
    exit 1
  fi
}

require_file "${TRAIN_INDEX}"
require_dir "${VLM_PATH}"

echo "[privileged-teacher] stage=${STAGE} bucket=${BUCKET_NAME} goal_mode=${GOAL_MODE}"
echo "[privileged-teacher] train_index=${TRAIN_INDEX}"
echo "[privileged-teacher] peer_indexes=${PEER_INDEXES_ARG}"
echo "[privileged-teacher] extra_metrics=${EXTRA_METRICS_ARG}"
echo "[privileged-teacher] bad_cache_list=${SCENE_ROUTER_BAD_CACHE_LIST}"

if [[ "${STAGE}" == "il" ]]; then
  require_file "${BASE_CKPT}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-teacher_${BUCKET_NAME}_goal_mixed_il_v2}"
  run_torch "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive_bucket_il.py" \
    "${COMMON[@]}" \
    "agent.checkpoint_path='${BASE_CKPT}'" \
    "agent.grpo=false" "agent.lr=${IL_LR}" \
    "bucket.full_ratio=${IL_FULL_RATIO}" "bucket.bucket_ratio=${IL_BUCKET_RATIO}" \
    "trainer.params.max_epochs=${IL_EPOCHS}" "dataloader.params.batch_size=${BATCH_SIZE_IL}" \
    "experiment_name='${EXPERIMENT_NAME}'"
elif [[ "${STAGE}" == "rl" ]]; then
  IL_CKPT="${IL_CKPT:?STAGE=rl requires IL_CKPT=old or new goal-aware IL checkpoint}"
  REFERENCE_POLICY_CKPT="${REFERENCE_POLICY_CKPT:-${IL_CKPT}}"
  require_file "${IL_CKPT}"
  require_file "${REFERENCE_POLICY_CKPT}"
  require_dir "${METRIC_CACHE_PATH}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-teacher_${BUCKET_NAME}_goal_bucket_rl_v2}"
  run_torch "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive_bucket_rl.py" \
    "${COMMON[@]}" \
    "agent.checkpoint_path='${IL_CKPT}'" \
    "agent.grpo=true" "agent.lr=${RL_LR}" \
    "agent.metric_cache_path='${METRIC_CACHE_PATH}'" \
    "agent.reference_policy_checkpoint='${REFERENCE_POLICY_CKPT}'" \
    "bucket.full_ratio=${RL_FULL_RATIO}" "bucket.bucket_ratio=${RL_BUCKET_RATIO}" \
    "trainer.params.max_epochs=${RL_EPOCHS}" "dataloader.params.batch_size=${BATCH_SIZE_RL}" \
    "experiment_name='${EXPERIMENT_NAME}'"
else
  echo "STAGE must be il or rl, got ${STAGE}" >&2
  exit 2
fi

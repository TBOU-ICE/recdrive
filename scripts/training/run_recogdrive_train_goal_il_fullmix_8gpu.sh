#!/usr/bin/env bash
# Stage 0 of self-distillation: full-mix goal-conditioned IL with goal corruption.
#
# Output: ONE checkpoint that is simultaneously
#   * a competent goal-free driver (goal masked 10% of the time), and
#   * a goal-follower that tolerates ~1m endpoint error (goal noised 20%).
# Stage 1 initialises both the teacher and the student pass from it.
#
# Deliberately NOT a bucket expert: the per-bucket privileged teachers scored 82
# EPDMS without a goal on general_or_no_tag, below the 87.4 the deployed student
# already reaches, which is why teacher->student distillation could not help.
#
# Data mixture is byte-identical to the GoalBridge OPD runs (navtrain full cache
# + SimScale quality rounds), so Stage 0 / Stage 1 / the old OPD baseline are
# comparable without any data confound.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export SCENE_ROUTER_BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${REPO_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"

NNODES="${NNODES:-1}"; RANK="${RANK:-0}"; GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-23531}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

# Warm start from the goal-free full-mix IL model: identical to how the student
# of every previous OPD run was initialised, so Stage 0 adds only the goal channel.
BASE_IL_CKPT="${BASE_IL_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"

VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/data/epdms/manifests}"
NAV_MANIFEST="${NAV_MANIFEST:-${MANIFEST_DIR}/nav_train_newvlm.json}"

USE_SIMSCALE="${USE_SIMSCALE:-1}"; SIM_ROUNDS="${SIM_ROUNDS:-0,1}"; SIM_REPEAT="${SIM_REPEAT:-1}"
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/workspace/datasets/simscale/20260709}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/simscale}"

GOAL_MODE="${GOAL_MODE:-adaln}"
# Disjoint partition: 70% clean / 20% noisy / 10% masked GT goal.
GOAL_DROPOUT_P="${GOAL_DROPOUT_P:-0.10}"
GOAL_NOISE_P="${GOAL_NOISE_P:-0.20}"
GOAL_NOISE_STD_XY="${GOAL_NOISE_STD_XY:-1.0}"
GOAL_NOISE_STD_HEADING="${GOAL_NOISE_STD_HEADING:-0.10}"

LR="${LR:-1e-4}"; MAX_EPOCHS="${MAX_EPOCHS:-30}"; BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

EXTRA_CACHE_LIST=(); EXTRA_REPEAT_LIST=(); EXTRA_TOKEN_JSON_LIST=(); EXTRA_MANIFEST_LIST=()
if [[ "${USE_SIMSCALE}" == "1" ]]; then
  IFS=',' read -r -a _rounds <<< "${SIM_ROUNDS}"
  for r in "${_rounds[@]}"; do
    r="${r//[[:space:]]/}"; [[ -z "${r}" ]] && continue
    ds="synthetic_reaction_pdm_v1.0-${r}"
    qjson="${SIMSCALE_BUCKET_ROOT}/scene_buckets_${ds}_quality/exclusive_token_to_bucket.json"
    qcache="${SIM_QUALITY_CACHE_ROOT}/recogdrive_agent_cache_dir_${ds}_quality"
    fcache="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${ds}"
    [[ -d "${qcache}" ]] && cache="${qcache}" || cache="${fcache}"
    manifest="${MANIFEST_DIR}/sim_round${r}_quality_newvlm.json"
    if [[ -d "${cache}" && -f "${qjson}" ]]; then
      EXTRA_CACHE_LIST+=("${cache}"); EXTRA_REPEAT_LIST+=("${SIM_REPEAT}")
      EXTRA_TOKEN_JSON_LIST+=("${qjson}"); EXTRA_MANIFEST_LIST+=("${manifest}")
    fi
  done
fi
join_hydra(){ if [[ $# -eq 0 ]]; then echo "[]"; else local IFS=,; echo "[$*]"; fi; }
EXTRA_CACHES_ARG="$(join_hydra ${EXTRA_CACHE_LIST[@]+"${EXTRA_CACHE_LIST[@]}"})"
EXTRA_REPEATS_ARG="$(join_hydra ${EXTRA_REPEAT_LIST[@]+"${EXTRA_REPEAT_LIST[@]}"})"
EXTRA_TOKEN_JSONS_ARG="$(join_hydra ${EXTRA_TOKEN_JSON_LIST[@]+"${EXTRA_TOKEN_JSON_LIST[@]}"})"
EXTRA_MANIFESTS_ARG="$(join_hydra ${EXTRA_MANIFEST_LIST[@]+"${EXTRA_MANIFEST_LIST[@]}"})"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_selfdistill_stage0_all_data_goal}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/run.log}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-/workspace/output/tensorboard}"
mkdir -p "$(dirname "${LOG_FILE}")" "${TENSORBOARD_DIR}"
HYDRA_RESUME=(); [[ -n "${RESUME_CKPT}" ]] && HYDRA_RESUME+=("+ckpt_path='${RESUME_CKPT}'")

echo "[Stage0] base=${BASE_IL_CKPT}"
echo "[Stage0] goal corruption: clean=$(python -c "print(1-${GOAL_DROPOUT_P}-${GOAL_NOISE_P})") masked=${GOAL_DROPOUT_P} noisy=${GOAL_NOISE_P}@${GOAL_NOISE_STD_XY}m"
echo "[Stage0] extra caches: ${EXTRA_CACHES_ARG}"

torchrun --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_goal_il_fullmix.py" \
  agent=recogdrive_agent_robust_goal_teacher \
  "agent.checkpoint_path='${BASE_IL_CKPT}'" "agent.vlm_path='${VLM_PATH}'" \
  agent.goal_mode="${GOAL_MODE}" \
  agent.goal_dropout_p="${GOAL_DROPOUT_P}" agent.goal_noise_p="${GOAL_NOISE_P}" \
  agent.goal_noise_std_xy="${GOAL_NOISE_STD_XY}" agent.goal_noise_std_heading="${GOAL_NOISE_STD_HEADING}" \
  agent.lr="${LR}" \
  "+scene_router_extra_cache_paths=${EXTRA_CACHES_ARG}" "+scene_router_extra_cache_repeats=${EXTRA_REPEATS_ARG}" \
  "+scene_router_extra_cache_token_json=${EXTRA_TOKEN_JSONS_ARG}" "+scene_router_cache_manifest='${NAV_MANIFEST}'" \
  "+scene_router_extra_cache_manifests=${EXTRA_MANIFESTS_ARG}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" trainer.params.precision=bf16-mixed trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" trainer.params.strategy=ddp_find_unused_parameters_true \
  "+tensorboard_dir='${TENSORBOARD_DIR}'" \
  dataloader.params.batch_size="${BATCH_SIZE}" dataloader.params.num_workers="${NUM_WORKERS}" dataloader.params.prefetch_factor=4 \
  +dataloader.params.persistent_workers=true experiment_name="${EXPERIMENT_NAME}" train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" use_cache_without_dataset=True force_cache_computation=False \
  ${HYDRA_RESUME[@]+"${HYDRA_RESUME[@]}"} hydra/job_logging=stdout hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

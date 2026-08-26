#!/usr/bin/env bash
# GoalBridge OPD v1: old GT-goal teachers + predicted-goal student.
# Student objective = goal recovery + recoverability-gated reverse-KL + base anchor.
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
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-23511}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

# Student can be the original goal-free IL checkpoint. Missing GoalBridge modules
# are intentionally initialised from scratch. Later you can point this at a
# GoalBridge checkpoint to continue training.
BASE_IL_CKPT="${BASE_IL_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
STUDENT_CKPT="${STUDENT_CKPT:-${BASE_IL_CKPT}}"
# Frozen goal-free deployable policy (new-vlm). Keep this fixed even when
# STUDENT_CKPT is later changed to a GoalBridge checkpoint for continuation.
ANCHOR_CKPT="${ANCHOR_CKPT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_scene_router_dit_goal_opd_v4/2026.08.24.06.12.01/lightning_logs/version_0/checkpoints/epoch=22-step=65228.ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"

# Old privileged teachers for the first run.
TEACHER_GOAL_MODE="${TEACHER_GOAL_MODE:-adaln}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_progress_curbside_stopgo_il_goal_adaln_newvlm/2026.08.04.10.54.30/lightning_logs/version_0/checkpoints/epoch=199-step=92800.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_rule_intersection_il_goal_adaln_newvlm/2026.08.02.16.39.57/lightning_logs/version_0/checkpoints/epoch=199-step=62800.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_safety_dynamics_interaction_il_goal_adaln_newvlm/2026.08.04.10.25.26/lightning_logs/version_0/checkpoints/epoch=199-step=38800.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_general_or_no_tag_il_goal_adaln_newvlm/2026.08.04.10.44.01/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"

VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
TOKEN_TO_BUCKET_JSON="${TOKEN_TO_BUCKET_JSON:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/data/epdms/manifests}"
NAV_MANIFEST="${NAV_MANIFEST:-${MANIFEST_DIR}/nav_train_newvlm.json}"

USE_SIMSCALE="${USE_SIMSCALE:-1}"; SIM_ROUNDS="${SIM_ROUNDS:-0,1}"; SIM_REPEAT="${SIM_REPEAT:-1}"
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/workspace/datasets/simscale/20260709}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/simscale}"

GOAL_LOSS_WEIGHT="${GOAL_LOSS_WEIGHT:-1.0}"
KD_WEIGHT="${KD_WEIGHT:-1.0}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.15}"
RECOVERABILITY_TAU_M="${RECOVERABILITY_TAU_M:-2.0}"
RECOVERABILITY_FLOOR="${RECOVERABILITY_FLOOR:-0.05}"
KL_PRECISION_CLIP="${KL_PRECISION_CLIP:-25.0}"
LR="${LR:-5e-5}"; MAX_EPOCHS="${MAX_EPOCHS:-50}"; BATCH_SIZE="${BATCH_SIZE:-8}"

TOKEN_JSON_LIST=("${TOKEN_TO_BUCKET_JSON}"); EXTRA_CACHE_LIST=(); EXTRA_REPEAT_LIST=(); EXTRA_TOKEN_JSON_LIST=(); EXTRA_MANIFEST_LIST=()
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
      TOKEN_JSON_LIST+=("${qjson}")
    fi
  done
fi
join_hydra(){ if [[ $# -eq 0 ]]; then echo "[]"; else local IFS=,; echo "[$*]"; fi; }
TOKEN_JSONS_ARG="$(join_hydra ${TOKEN_JSON_LIST[@]+"${TOKEN_JSON_LIST[@]}"})"
EXTRA_CACHES_ARG="$(join_hydra ${EXTRA_CACHE_LIST[@]+"${EXTRA_CACHE_LIST[@]}"})"
EXTRA_REPEATS_ARG="$(join_hydra ${EXTRA_REPEAT_LIST[@]+"${EXTRA_REPEAT_LIST[@]}"})"
EXTRA_TOKEN_JSONS_ARG="$(join_hydra ${EXTRA_TOKEN_JSON_LIST[@]+"${EXTRA_TOKEN_JSON_LIST[@]}"})"
EXTRA_MANIFESTS_ARG="$(join_hydra ${EXTRA_MANIFEST_LIST[@]+"${EXTRA_MANIFEST_LIST[@]}"})"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_goalbridge_opd_old_teacher_gpt_rl_anchor_v1}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/run.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
HYDRA_RESUME=(); [[ -n "${RESUME_CKPT}" ]] && HYDRA_RESUME+=("+ckpt_path='${RESUME_CKPT}'")

echo "[GoalBridge] student=${STUDENT_CKPT} old_teachers=true goal_w=${GOAL_LOSS_WEIGHT} kd_w=${KD_WEIGHT} anchor_w=${ANCHOR_WEIGHT} tau=${RECOVERABILITY_TAU_M}m"

torchrun --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_scene_router_dit_goal_distill.py" \
  agent=recogdrive_agent_goalbridge_opd \
  "agent.checkpoint_path='${STUDENT_CKPT}'" "agent.anchor_checkpoint='${ANCHOR_CKPT}'" "agent.vlm_path='${VLM_PATH}'" \
  "agent.teacher_goal_mode='${TEACHER_GOAL_MODE}'" \
  "agent.teacher_ckpt_progress_curbside_stopgo='${TEACHER_PROGRESS_CKPT}'" \
  "agent.teacher_ckpt_rule_intersection='${TEACHER_RULE_CKPT}'" \
  "agent.teacher_ckpt_safety_dynamics_interaction='${TEACHER_SAFETY_CKPT}'" \
  "agent.teacher_ckpt_general_or_no_tag='${TEACHER_GENERAL_CKPT}'" \
  "agent.token_to_bucket_json=${TOKEN_JSONS_ARG}" \
  "+scene_router_extra_cache_paths=${EXTRA_CACHES_ARG}" "+scene_router_extra_cache_repeats=${EXTRA_REPEATS_ARG}" \
  "+scene_router_extra_cache_token_json=${EXTRA_TOKEN_JSONS_ARG}" "+scene_router_cache_manifest='${NAV_MANIFEST}'" \
  "+scene_router_extra_cache_manifests=${EXTRA_MANIFESTS_ARG}" \
  agent.goal_loss_weight="${GOAL_LOSS_WEIGHT}" agent.kd_weight="${KD_WEIGHT}" agent.anchor_weight="${ANCHOR_WEIGHT}" \
  agent.recoverability_tau_m="${RECOVERABILITY_TAU_M}" agent.recoverability_floor="${RECOVERABILITY_FLOOR}" \
  agent.kl_precision_clip="${KL_PRECISION_CLIP}" agent.lr="${LR}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" trainer.params.precision=bf16-mixed trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size="${BATCH_SIZE}" dataloader.params.num_workers=8 dataloader.params.prefetch_factor=4 \
  +dataloader.params.persistent_workers=true experiment_name="${EXPERIMENT_NAME}" train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" use_cache_without_dataset=True force_cache_computation=False \
  ${HYDRA_RESUME[@]+"${HYDRA_RESUME[@]}"} hydra/job_logging=stdout hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

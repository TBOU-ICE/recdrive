#!/usr/bin/env bash
# 8-GPU scene-router GOAL-teacher DiT OPD -- RESEARCH variants A/B harness.
#   Select one variant via VARIANT env (default: kl):
#     VARIANT=kl      方案一 logprob-KL  (match transition mean mu)
#     VARIANT=anchor  方案二 Manifold Anchor 正则 (pull final x0 to frozen base)
#     VARIANT=phf     方案三 PHF 特权鸿沟 (align DiT hidden + flow to teacher)
#
#   teachers        = four goal-conditioned IL experts (adaln, epoch=199)
#   student         = plain goal-free DiT (privileged info is distilled)
#   teacher_select  = scene_route ; exopd_lambda = 1.0 (pure OPD)
#
# Additive: does NOT modify any existing file. Uses the research agent config
#   recogdrive_agent_scene_router_dit_goal_distill_${VARIANT}.yaml
# so the baseline goal-distill script is completely untouched.
set -euo pipefail

VARIANT="${VARIANT:-kl}"
case "${VARIANT}" in
  kl|anchor|phf) ;;
  *) echo "ERROR: VARIANT must be one of {kl,anchor,phf}, got '${VARIANT}'" >&2; exit 1 ;;
esac
AGENT_CFG="recogdrive_agent_scene_router_dit_goal_distill_${VARIANT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export SCENE_ROUTER_BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${REPO_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23471}"
GPUS="${GPUS:-8}"

# ---- student init = new-vlm IL base (representation-consistent) ----
STUDENT_CKPT="${STUDENT_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"

# ---- four goal-conditioned scenario teachers (adaln IL experts, epoch=199) ----
TEACHER_GOAL_MODE="${TEACHER_GOAL_MODE:-adaln}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_progress_curbside_stopgo_il_goal_adaln_newvlm/2026.08.04.10.54.30/lightning_logs/version_0/checkpoints/epoch=199-step=92800.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_rule_intersection_il_goal_adaln_newvlm/2026.08.02.16.39.57/lightning_logs/version_0/checkpoints/epoch=199-step=62800.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_safety_dynamics_interaction_il_goal_adaln_newvlm/2026.08.04.10.25.26/lightning_logs/version_0/checkpoints/epoch=199-step=38800.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_general_or_no_tag_il_goal_adaln_newvlm/2026.08.04.10.44.01/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"

# ---- pure OPD (target == goal-conditioned expert); ref unused at lambda=1.0 ----
EXOPD_LAMBDA="${EXOPD_LAMBDA:-1.0}"
# match_target defaults per-variant from the config (mu for kl, x0 otherwise);
# only override if the caller explicitly sets MATCH_TARGET.
MATCH_TARGET="${MATCH_TARGET:-}"

# ---- variant weights (fall back to the config values when unset) ----
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-}"
PHF_WEIGHT="${PHF_WEIGHT:-}"
PHF_FLOW_WEIGHT="${PHF_FLOW_WEIGHT:-}"
ANCHOR_CKPT="${ANCHOR_CKPT:-}"

# ---- token -> scenario bucket map ----
TOKEN_TO_BUCKET_JSON="${TOKEN_TO_BUCKET_JSON:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"

# ---- new-VLM representation + cache (must match the teachers!) ----
VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"

# ---- prebuilt token indexes (skip multi-hour Alluxio cache walks) ----
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/data/epdms/manifests}"
NAV_MANIFEST="${NAV_MANIFEST:-${MANIFEST_DIR}/nav_train_newvlm.json}"

# ---- simscale mix ----
USE_SIMSCALE="${USE_SIMSCALE:-1}"
SIM_ROUNDS="${SIM_ROUNDS:-0,1}"
SIM_REPEAT="${SIM_REPEAT:-1}"
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/workspace/datasets/simscale/20260709}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/simscale}"

TOKEN_JSON_LIST=("${TOKEN_TO_BUCKET_JSON}")
EXTRA_CACHE_LIST=()
EXTRA_REPEAT_LIST=()
EXTRA_TOKEN_JSON_LIST=()
EXTRA_MANIFEST_LIST=()
if [[ "${USE_SIMSCALE}" == "1" ]]; then
  IFS=',' read -r -a _sim_rounds <<< "${SIM_ROUNDS}"
  for r in "${_sim_rounds[@]}"; do
    r="${r//[[:space:]]/}"; [[ -z "${r}" ]] && continue
    ds="synthetic_reaction_pdm_v1.0-${r}"
    qjson="${SIMSCALE_BUCKET_ROOT}/scene_buckets_${ds}_quality/exclusive_token_to_bucket.json"
    qcache="${SIM_QUALITY_CACHE_ROOT}/recogdrive_agent_cache_dir_${ds}_quality"
    fcache="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${ds}"
    if [[ -d "${qcache}" ]]; then cache="${qcache}"; elif [[ -d "${fcache}" ]]; then cache="${fcache}"; else cache=""; fi
    manifest="${MANIFEST_DIR}/sim_round${r}_quality_newvlm.json"
    if [[ -n "${cache}" && -f "${qjson}" ]]; then
      EXTRA_CACHE_LIST+=("${cache}")
      EXTRA_REPEAT_LIST+=("${SIM_REPEAT}")
      EXTRA_TOKEN_JSON_LIST+=("${qjson}")
      EXTRA_MANIFEST_LIST+=("${manifest}")
      TOKEN_JSON_LIST+=("${qjson}")
      echo "[research-${VARIANT}] + simscale round ${r}: cache=${cache} manifest=${manifest}"
    else
      echo "[research-${VARIANT}] ! skip simscale round ${r} (missing cache or quality bucket json)"
    fi
  done
fi
join_hydra() { if [[ $# -eq 0 ]]; then echo "[]"; else local IFS=,; echo "[$*]"; fi; }
TOKEN_JSONS_ARG="$(join_hydra ${TOKEN_JSON_LIST[@]+"${TOKEN_JSON_LIST[@]}"})"
EXTRA_CACHES_ARG="$(join_hydra ${EXTRA_CACHE_LIST[@]+"${EXTRA_CACHE_LIST[@]}"})"
EXTRA_REPEATS_ARG="$(join_hydra ${EXTRA_REPEAT_LIST[@]+"${EXTRA_REPEAT_LIST[@]}"})"
EXTRA_TOKEN_JSONS_ARG="$(join_hydra ${EXTRA_TOKEN_JSON_LIST[@]+"${EXTRA_TOKEN_JSON_LIST[@]}"})"
EXTRA_MANIFESTS_ARG="$(join_hydra ${EXTRA_MANIFEST_LIST[@]+"${EXTRA_MANIFEST_LIST[@]}"})"

if [[ ! -f "${NAV_MANIFEST}" ]]; then
  echo "[research-${VARIANT}] ERROR: nav manifest missing: ${NAV_MANIFEST}" >&2
  exit 1
fi
for m in "${EXTRA_MANIFEST_LIST[@]+"${EXTRA_MANIFEST_LIST[@]}"}"; do
  if [[ ! -f "${m}" ]]; then
    echo "[research-${VARIANT}] ERROR: sim manifest missing: ${m}" >&2
    exit 1
  fi
done

MAX_EPOCHS="${MAX_EPOCHS:-35}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_scene_router_dit_goal_opd_${VARIANT}_v1}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/run_scene_router_dit_goal_opd_${VARIANT}.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "[research-${VARIANT}] AGENT_CFG=${AGENT_CFG} GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "[research-${VARIANT}] max_epochs=${MAX_EPOCHS} exopd_lambda=${EXOPD_LAMBDA}"
echo "[research-${VARIANT}] STUDENT_CKPT=${STUDENT_CKPT}"
echo "[research-${VARIANT}] LOG_FILE=${LOG_FILE}"

# optional variant overrides (only appended when the caller set them)
EXTRA_OVERRIDES=()
[[ -n "${MATCH_TARGET}" ]]    && EXTRA_OVERRIDES+=("agent.match_target='${MATCH_TARGET}'")
[[ -n "${ANCHOR_WEIGHT}" ]]   && EXTRA_OVERRIDES+=("agent.anchor_weight=${ANCHOR_WEIGHT}")
[[ -n "${PHF_WEIGHT}" ]]      && EXTRA_OVERRIDES+=("agent.phf_weight=${PHF_WEIGHT}")
[[ -n "${PHF_FLOW_WEIGHT}" ]] && EXTRA_OVERRIDES+=("agent.phf_flow_weight=${PHF_FLOW_WEIGHT}")
[[ -n "${ANCHOR_CKPT}" ]]     && EXTRA_OVERRIDES+=("agent.anchor_checkpoint='${ANCHOR_CKPT}'")

/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_scene_router_dit_goal_distill.py" \
  agent="${AGENT_CFG}" \
  "agent.checkpoint_path='${STUDENT_CKPT}'" \
  "agent.vlm_path='${VLM_PATH}'" \
  "agent.teacher_goal_mode='${TEACHER_GOAL_MODE}'" \
  "agent.teacher_ckpt_progress_curbside_stopgo='${TEACHER_PROGRESS_CKPT}'" \
  "agent.teacher_ckpt_rule_intersection='${TEACHER_RULE_CKPT}'" \
  "agent.teacher_ckpt_safety_dynamics_interaction='${TEACHER_SAFETY_CKPT}'" \
  "agent.teacher_ckpt_general_or_no_tag='${TEACHER_GENERAL_CKPT}'" \
  "agent.token_to_bucket_json=${TOKEN_JSONS_ARG}" \
  "+scene_router_extra_cache_paths=${EXTRA_CACHES_ARG}" \
  "+scene_router_extra_cache_repeats=${EXTRA_REPEATS_ARG}" \
  "+scene_router_extra_cache_token_json=${EXTRA_TOKEN_JSONS_ARG}" \
  "+scene_router_cache_manifest='${NAV_MANIFEST}'" \
  "+scene_router_extra_cache_manifests=${EXTRA_MANIFESTS_ARG}" \
  agent.teacher_select='scene_route' \
  agent.exopd_lambda="${EXOPD_LAMBDA}" \
  agent.scene_router_smooth_weight=0.02 \
  agent.scene_router_min_sigma=0.04 \
  agent.viz_interval_steps=0 \
  agent.lr=1e-4 \
  agent.grpo=False \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size=8 \
  dataloader.params.num_workers=8 \
  dataloader.params.prefetch_factor=4 \
  +dataloader.params.persistent_workers=true \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

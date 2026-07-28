#!/usr/bin/env bash
# 8-GPU scene-router four-teacher DiT OPD (v1).
#   teacher_select=scene_route (per-token bucket -> single expert)
#   match_target=x0            (match predicted clean trajectory, eta-independent)
#   exopd_lambda=1.25          (reward extrapolation: target = ref + λ(expert - ref))
# Additive: does not modify any existing file.
#
# Representation contract: CACHE_PATH / VLM_PATH must be the new-VLM
# (vlm_simscale_lora_merged) representation that produced the teachers. Do NOT
# point at old-VLM or *_vit_* caches, or the DiT reads OOD features.
set -euo pipefail

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/code}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23461}"
GPUS="${GPUS:-8}"

# ---- student init = new-vlm IL base (representation-consistent; == ExOPD ref) ----
# NOTE: the new-vlm IL run only saved epoch=194/199; there is no early new-vlm ckpt.
# The old-vlm training_recogdrive_vlm_il/epoch=2 is a DIFFERENT representation - do not use.
STUDENT_CKPT="${STUDENT_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"

# ---- four scenario-expert teachers (new-vlm RL experts) ----
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_teacher_progress_curbside_stopgo_rl_newvlm/2026.07.24.05.09.01/lightning_logs/version_0/checkpoints/ckpt/epoch=14-step=2385.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_teacher_rule_intersection_rl_newvlm/2026.07.24.04.59.12/lightning_logs/version_0/checkpoints/ckpt/epoch=14-step=2115.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_teacher_safety_dynamics_interaction_rl_newvlm/2026.07.27.07.05.26/lightning_logs/version_0/checkpoints/epoch=14-step=1890.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_teacher_general_or_no_tag_rl_newvlm/2026.07.23.22.12.41/lightning_logs/version_0/checkpoints/epoch=11-step=2100.ckpt}"

# ---- ExOPD reference (experts' pre-RL IL base, new-vlm 199-epoch) ----
EXOPD_REF_CKPT="${EXOPD_REF_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
EXOPD_LAMBDA="${EXOPD_LAMBDA:-1.25}"
MATCH_TARGET="${MATCH_TARGET:-x0}"

# ---- token -> scenario bucket map ----
TOKEN_TO_BUCKET_JSON="${TOKEN_TO_BUCKET_JSON:-/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"

# ---- new-VLM representation + cache (must match the teachers!) ----
VLM_PATH="${VLM_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/vlm_simscale_lora_merged}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"

# ---- simscale mix (the experts were RL'd with ~0.4 simscale-bucket; on-policy OPD
#      needs the student to visit those scenario states, so we add simscale here too) ----
USE_SIMSCALE="${USE_SIMSCALE:-1}"
SIM_ROUNDS="${SIM_ROUNDS:-0,1}"     # comma list e.g. "0" or "0,1"; each round is token-filtered by its _quality bucket json
SIM_REPEAT="${SIM_REPEAT:-1}"       # up-weight each simscale round by including it SIM_REPEAT times (DDP-safe)
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
# Pre-built *_quality symlink views (both round0 AND round1) live here. Each view symlinks into
# the FULL caches under SIM_AGENT_CACHE_ROOT (/workspace/datasets -> /mnt/datasets), so its round0
# is byte-identical to round0 under SIM_AGENT_CACHE_ROOT; unlike that root, it also has round1_quality.
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale/new_vlm_quality_views}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"

TOKEN_JSON_LIST=("${TOKEN_TO_BUCKET_JSON}")   # routing map: navtrain + simscale rounds (merged)
EXTRA_CACHE_LIST=()                            # simscale agent caches (extra streams)
EXTRA_REPEAT_LIST=()
EXTRA_TOKEN_JSON_LIST=()                        # per-extra-cache token whitelist (== that round's quality bucket json)
if [[ "${USE_SIMSCALE}" == "1" ]]; then
  IFS=',' read -r -a _sim_rounds <<< "${SIM_ROUNDS}"
  for r in "${_sim_rounds[@]}"; do
    r="${r//[[:space:]]/}"; [[ -z "${r}" ]] && continue
    ds="synthetic_reaction_pdm_v1.0-${r}"
    qjson="${SIMSCALE_BUCKET_ROOT}/scene_buckets_${ds}_quality/exclusive_token_to_bucket.json"
    qcache="${SIM_QUALITY_CACHE_ROOT}/recogdrive_agent_cache_dir_${ds}_quality"
    fcache="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${ds}"
    # prefer a *_quality agent cache (symlink view); else use the FULL cache token-filtered by the quality bucket json
    if [[ -d "${qcache}" ]]; then cache="${qcache}"; elif [[ -d "${fcache}" ]]; then cache="${fcache}"; else cache=""; fi
    if [[ -n "${cache}" && -f "${qjson}" ]]; then
      EXTRA_CACHE_LIST+=("${cache}")
      EXTRA_REPEAT_LIST+=("${SIM_REPEAT}")
      EXTRA_TOKEN_JSON_LIST+=("${qjson}")
      TOKEN_JSON_LIST+=("${qjson}")
      echo "[scene-router-v1] + simscale round ${r}: cache=${cache} (token-filtered by quality bucket json)"
    else
      echo "[scene-router-v1] ! skip simscale round ${r} (missing cache or quality bucket json)"
    fi
  done
fi
join_hydra() { if [[ $# -eq 0 ]]; then echo "[]"; else local IFS=,; echo "[$*]"; fi; }
TOKEN_JSONS_ARG="$(join_hydra ${TOKEN_JSON_LIST[@]+"${TOKEN_JSON_LIST[@]}"})"
EXTRA_CACHES_ARG="$(join_hydra ${EXTRA_CACHE_LIST[@]+"${EXTRA_CACHE_LIST[@]}"})"
EXTRA_REPEATS_ARG="$(join_hydra ${EXTRA_REPEAT_LIST[@]+"${EXTRA_REPEAT_LIST[@]}"})"
EXTRA_TOKEN_JSONS_ARG="$(join_hydra ${EXTRA_TOKEN_JSON_LIST[@]+"${EXTRA_TOKEN_JSON_LIST[@]}"})"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_scene_router_dit_opd_v1}"

echo "[scene-router-v1] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[scene-router-v1] teacher_select=scene_route match_target=${MATCH_TARGET} exopd_lambda=${EXOPD_LAMBDA}"
echo "[scene-router-v1] STUDENT_CKPT=${STUDENT_CKPT}"
echo "[scene-router-v1] EXOPD_REF_CKPT=${EXOPD_REF_CKPT}"
echo "[scene-router-v1] CACHE_PATH=${CACHE_PATH}"
echo "[scene-router-v1] TOKEN_TO_BUCKET_JSON=${TOKEN_TO_BUCKET_JSON}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_scene_router_dit_distill.py" \
  agent=recogdrive_agent_scene_router_dit_distill \
  "agent.checkpoint_path='${STUDENT_CKPT}'" \
  "agent.vlm_path='${VLM_PATH}'" \
  "agent.teacher_ckpt_progress_curbside_stopgo='${TEACHER_PROGRESS_CKPT}'" \
  "agent.teacher_ckpt_rule_intersection='${TEACHER_RULE_CKPT}'" \
  "agent.teacher_ckpt_safety_dynamics_interaction='${TEACHER_SAFETY_CKPT}'" \
  "agent.teacher_ckpt_general_or_no_tag='${TEACHER_GENERAL_CKPT}'" \
  "agent.token_to_bucket_json=${TOKEN_JSONS_ARG}" \
  "+scene_router_extra_cache_paths=${EXTRA_CACHES_ARG}" \
  "+scene_router_extra_cache_repeats=${EXTRA_REPEATS_ARG}" \
  "+scene_router_extra_cache_token_json=${EXTRA_TOKEN_JSONS_ARG}" \
  agent.teacher_select='scene_route' \
  "agent.match_target='${MATCH_TARGET}'" \
  "agent.exopd_ref_checkpoint='${EXOPD_REF_CKPT}'" \
  agent.exopd_lambda="${EXOPD_LAMBDA}" \
  agent.scene_router_smooth_weight=0.02 \
  agent.scene_router_min_sigma=0.04 \
  agent.lr=1e-4 \
  agent.grpo=False \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  trainer.params.max_epochs=50 \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size=8 \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

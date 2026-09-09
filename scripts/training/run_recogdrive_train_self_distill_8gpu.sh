#!/usr/bin/env bash
# Stage 1 of self-distillation: one model, two goal conditions, no teacher ckpt.
#
# Each batch runs the SAME weights twice on the SAME on-policy DDIM states:
#   teacher pass -> conditioned on the GT endpoint (privileged)
#   student pass -> conditioned on its own predicted endpoint (deployable)
# loss = il_weight * il(teacher pass, GT traj)
#      + kd_weight * reverse_KL(student pass || teacher pass)
#      + goal_loss_weight * smooth_l1(pred_goal, GT goal)
#
# STUDENT_CKPT must be the Stage 0 checkpoint. Do NOT start from an old
# teacher-OPD student: that model's competence is inherited from the old
# privileged teachers, so any gain here would be unattributable.
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
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-23541}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

STUDENT_CKPT="${STUDENT_CKPT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_selfdistill_stage0_all_data_goal/2026.09.08.14.39.46/checkpoints/epoch=29-step=42540.ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"

VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
TOKEN_TO_BUCKET_JSON="${TOKEN_TO_BUCKET_JSON:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/data/epdms/manifests}"
NAV_MANIFEST="${NAV_MANIFEST:-${MANIFEST_DIR}/nav_train_newvlm.json}"

USE_SIMSCALE="${USE_SIMSCALE:-1}"; SIM_ROUNDS="${SIM_ROUNDS:-0,1}"; SIM_REPEAT="${SIM_REPEAT:-1}"
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/workspace/datasets/simscale/20260709}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/simscale}"

IL_WEIGHT="${IL_WEIGHT:-1.0}"
KD_WEIGHT="${KD_WEIGHT:-1.0}"
GOAL_LOSS_WEIGHT="${GOAL_LOSS_WEIGHT:-1.0}"
TEACHER_GOAL_NOISE_P="${TEACHER_GOAL_NOISE_P:-0.2}"
TEACHER_GOAL_NOISE_STD_XY="${TEACHER_GOAL_NOISE_STD_XY:-1.0}"
TEACHER_GOAL_DROPOUT_P="${TEACHER_GOAL_DROPOUT_P:-0.0}"
GOAL_DETACH_ENCODERS="${GOAL_DETACH_ENCODERS:-False}"
KL_PRECISION_CLIP="${KL_PRECISION_CLIP:-25.0}"
GOAL_PROBE_INTERVAL="${GOAL_PROBE_INTERVAL:-50}"
COLLECT_VIZ="${COLLECT_VIZ:-1}"
VIZ_INTERVAL_STEPS="${VIZ_INTERVAL_STEPS:-500}"
# Two grad-carrying DiT passes per DDIM step; halve this vs GoalBridge OPD if OOM.
LR="${LR:-5e-5}"; MAX_EPOCHS="${MAX_EPOCHS:-30}"; BATCH_SIZE="${BATCH_SIZE:-8}"

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
EXTRA_CACHES_ARG="$(join_hydra ${EXTRA_CACHE_LIST[@]+"${EXTRA_CACHE_LIST[@]}"})"
EXTRA_REPEATS_ARG="$(join_hydra ${EXTRA_REPEAT_LIST[@]+"${EXTRA_REPEAT_LIST[@]}"})"
EXTRA_TOKEN_JSONS_ARG="$(join_hydra ${EXTRA_TOKEN_JSON_LIST[@]+"${EXTRA_TOKEN_JSON_LIST[@]}"})"
EXTRA_MANIFESTS_ARG="$(join_hydra ${EXTRA_MANIFEST_LIST[@]+"${EXTRA_MANIFEST_LIST[@]}"})"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_selfdistill_stage1_v1}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/run.log}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-/workspace/output/tensorboard}"
mkdir -p "$(dirname "${LOG_FILE}")" "${TENSORBOARD_DIR}"
HYDRA_RESUME=(); [[ -n "${RESUME_CKPT}" ]] && HYDRA_RESUME+=("+ckpt_path='${RESUME_CKPT}'")

echo "[SelfDistill] stage0_ckpt=${STUDENT_CKPT}"
echo "[SelfDistill] il_w=${IL_WEIGHT} kd_w=${KD_WEIGHT} goal_w=${GOAL_LOSS_WEIGHT}"
echo "[SelfDistill] teacher goal noise ${TEACHER_GOAL_NOISE_P}@${TEACHER_GOAL_NOISE_STD_XY}m, masked ${TEACHER_GOAL_DROPOUT_P}"

torchrun --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_scene_router_dit_goal_distill.py" \
  agent=recogdrive_agent_self_distill \
  "agent.checkpoint_path='${STUDENT_CKPT}'" "agent.vlm_path='${VLM_PATH}'" \
  agent.il_weight="${IL_WEIGHT}" agent.kd_weight="${KD_WEIGHT}" agent.goal_loss_weight="${GOAL_LOSS_WEIGHT}" \
  agent.teacher_goal_noise_p="${TEACHER_GOAL_NOISE_P}" agent.teacher_goal_noise_std_xy="${TEACHER_GOAL_NOISE_STD_XY}" \
  agent.teacher_goal_dropout_p="${TEACHER_GOAL_DROPOUT_P}" \
  agent.goal_detach_encoders="${GOAL_DETACH_ENCODERS}" \
  agent.kl_precision_clip="${KL_PRECISION_CLIP}" agent.goal_probe_interval="${GOAL_PROBE_INTERVAL}" \
  agent.lr="${LR}" agent.collect_viz="${COLLECT_VIZ}" agent.viz_interval_steps="${VIZ_INTERVAL_STEPS}" \
  "+scene_router_extra_cache_paths=${EXTRA_CACHES_ARG}" "+scene_router_extra_cache_repeats=${EXTRA_REPEATS_ARG}" \
  "+scene_router_extra_cache_token_json=${EXTRA_TOKEN_JSONS_ARG}" "+scene_router_cache_manifest='${NAV_MANIFEST}'" \
  "+scene_router_extra_cache_manifests=${EXTRA_MANIFESTS_ARG}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" trainer.params.precision=bf16-mixed trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" trainer.params.strategy=ddp_find_unused_parameters_true \
  "+tensorboard_dir='${TENSORBOARD_DIR}'" \
  dataloader.params.batch_size="${BATCH_SIZE}" dataloader.params.num_workers=8 dataloader.params.prefetch_factor=4 \
  +dataloader.params.persistent_workers=true experiment_name="${EXPERIMENT_NAME}" train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" use_cache_without_dataset=True force_cache_computation=False \
  ${HYDRA_RESUME[@]+"${HYDRA_RESUME[@]}"} hydra/job_logging=stdout hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

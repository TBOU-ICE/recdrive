#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need CONDA_BIN; need NAVSIM_EXP_ROOT
need STUDENT_CKPT; need VLM_PATH; need NAV_CACHE; need NAV_MANIFEST; need TOKEN_TO_BUCKET_NAV

TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-${PRIV_PROGRESS_CKPT:-}}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-${PRIV_RULE_CKPT:-}}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-${PRIV_SAFETY_CKPT:-}}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-${PRIV_GENERAL_CKPT:-}}"
need TEACHER_PROGRESS_CKPT; need TEACHER_RULE_CKPT; need TEACHER_SAFETY_CKPT; need TEACHER_GENERAL_CKPT

VARIANT="${VARIANT:-A}"
GOAL_INJECTION="${GOAL_INJECTION:-gated_cross}"
GOAL_POINT_MODE="${GOAL_POINT_MODE:-multi3}"
GOAL_INDICES="${GOAL_INDICES:-[1,4,7]}"
KD_WEIGHT="${KD_WEIGHT:-1.0}"
TASK_WEIGHT="${TASK_WEIGHT:-0.10}"
GOAL_AUX_WEIGHT="${GOAL_AUX_WEIGHT:-0.0}"
GOAL_PREF_WEIGHT="${GOAL_PREF_WEIGHT:-0.0}"
GOAL_PREF_TAU_M="${GOAL_PREF_TAU_M:-2.0}"
GOAL_PREF_CANDIDATES="${GOAL_PREF_CANDIDATES:-8}"
GOAL_PREF_GEO_WEIGHT="${GOAL_PREF_GEO_WEIGHT:-0.25}"
RESIDUAL_PRIVILEGE="${RESIDUAL_PRIVILEGE:-false}"
RESIDUAL_REF_CKPT="${RESIDUAL_REF_CKPT:-$STUDENT_CKPT}"
PRECISION_CLIP="${PRECISION_CLIP:-25.0}"
SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.0}"
LR="${LR:-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_DIAG_INTERVAL="${GRAD_DIAG_INTERVAL:-100}"
GPUS="${GPUS:-8}"; NNODES="${NNODES:-1}"; NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-23701}"
USE_SIMSCALE="${USE_SIMSCALE:-1}"; SIM_REPEAT="${SIM_REPEAT:-1}"

TOKEN_JSONS=("$TOKEN_TO_BUCKET_NAV")
EXTRA_PATHS=(); EXTRA_MANIFESTS=(); EXTRA_TOKEN_JSONS=(); EXTRA_REPEATS=()
if [[ "$USE_SIMSCALE" == "1" ]]; then
  for r in 0 1; do
    eval cache="\${SIM_CACHE_R${r}:-}"
    eval manifest="\${SIM_MANIFEST_R${r}:-}"
    eval root="\${SIM_BUCKET_R${r}_ROOT:-}"
    route_json="${root:-}/exclusive_token_to_bucket.json"
    if [[ -n "${cache:-}" && -d "$cache" && -n "${manifest:-}" && -f "$manifest" && -f "$route_json" ]]; then
      EXTRA_PATHS+=("$cache"); EXTRA_MANIFESTS+=("$manifest"); EXTRA_TOKEN_JSONS+=("$route_json"); EXTRA_REPEATS+=("$SIM_REPEAT"); TOKEN_JSONS+=("$route_json")
    fi
  done
fi
join(){ if [[ $# -eq 0 ]]; then echo '[]'; else local IFS=,; echo "[$*]"; fi; }
TOKEN_JSONS_ARG="$(join ${TOKEN_JSONS[@]+"${TOKEN_JSONS[@]}"})"
EXTRA_PATHS_ARG="$(join ${EXTRA_PATHS[@]+"${EXTRA_PATHS[@]}"})"
EXTRA_MANIFESTS_ARG="$(join ${EXTRA_MANIFESTS[@]+"${EXTRA_MANIFESTS[@]}"})"
EXTRA_TOKEN_JSONS_ARG="$(join ${EXTRA_TOKEN_JSONS[@]+"${EXTRA_TOKEN_JSONS[@]}"})"
EXTRA_REPEATS_ARG="$(join ${EXTRA_REPEATS[@]+"${EXTRA_REPEATS[@]}"})"

GOAL_VOCAB_ARG="null"
if [[ "$GOAL_PREF_WEIGHT" != "0" && "$GOAL_PREF_WEIGHT" != "0.0" ]]; then
  need GOAL_VOCAB; GOAL_VOCAB_ARG="'$GOAL_VOCAB'"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_privileged_opd_v2_${VARIANT}_${GOAL_POINT_MODE}_${GOAL_INJECTION}}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/run.log}"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[OPD-v2/$VARIANT] route=ONE teacher/sample; student=$STUDENT_CKPT"
echo "[OPD-v2/$VARIANT] teacher_arch=$GOAL_POINT_MODE/$GOAL_INJECTION KD=$KD_WEIGHT task=$TASK_WEIGHT aux=$GOAL_AUX_WEIGHT pref=$GOAL_PREF_WEIGHT residual=$RESIDUAL_PRIVILEGE"

"${CONDA_BIN}/torchrun" \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_privileged_opd_v2.py" \
  agent=recogdrive_agent_privileged_opd_v2 \
  "agent.checkpoint_path='$STUDENT_CKPT'" "agent.vlm_path='$VLM_PATH'" \
  "agent.teacher_ckpt_progress_curbside_stopgo='$TEACHER_PROGRESS_CKPT'" \
  "agent.teacher_ckpt_rule_intersection='$TEACHER_RULE_CKPT'" \
  "agent.teacher_ckpt_safety_dynamics_interaction='$TEACHER_SAFETY_CKPT'" \
  "agent.teacher_ckpt_general_or_no_tag='$TEACHER_GENERAL_CKPT'" \
  "agent.token_to_bucket_json=$TOKEN_JSONS_ARG" \
  "agent.teacher_goal_injection='$GOAL_INJECTION'" "agent.teacher_goal_point_mode='$GOAL_POINT_MODE'" "agent.teacher_goal_indices=$GOAL_INDICES" \
  agent.kd_weight="$KD_WEIGHT" agent.task_weight="$TASK_WEIGHT" agent.goal_aux_weight="$GOAL_AUX_WEIGHT" \
  agent.goal_pref_weight="$GOAL_PREF_WEIGHT" agent.goal_pref_temperature_m="$GOAL_PREF_TAU_M" \
  agent.goal_pref_candidate_count="$GOAL_PREF_CANDIDATES" agent.goal_pref_geo_weight="$GOAL_PREF_GEO_WEIGHT" \
  "agent.goal_vocab_path=$GOAL_VOCAB_ARG" agent.residual_privilege="$RESIDUAL_PRIVILEGE" \
  "agent.residual_ref_checkpoint='$RESIDUAL_REF_CKPT'" agent.precision_clip="$PRECISION_CLIP" \
  agent.scene_router_smooth_weight="$SMOOTH_WEIGHT" agent.lr="$LR" agent.train_epochs="$MAX_EPOCHS" \
  "+opd_v2_nav_manifest='$NAV_MANIFEST'" "+opd_v2_extra_cache_paths=$EXTRA_PATHS_ARG" \
  "+opd_v2_extra_cache_manifests=$EXTRA_MANIFESTS_ARG" "+opd_v2_extra_cache_token_json=$EXTRA_TOKEN_JSONS_ARG" \
  "+opd_v2_extra_cache_repeats=$EXTRA_REPEATS_ARG" +opd_v2_grad_diag_interval="$GRAD_DIAG_INTERVAL" \
  trainer.params.max_epochs="$MAX_EPOCHS" trainer.params.precision=bf16-mixed trainer.params.num_nodes="$NNODES" trainer.params.devices="$GPUS" \
  trainer.params.strategy=ddp_find_unused_parameters_true trainer.params.check_val_every_n_epoch=1 trainer.params.num_sanity_val_steps=0 \
  dataloader.params.batch_size="$BATCH_SIZE" dataloader.params.num_workers=8 dataloader.params.prefetch_factor=4 +dataloader.params.persistent_workers=true \
  experiment_name="$EXPERIMENT_NAME" train_test_split=navtrain cache_path="$NAV_CACHE" use_cache_without_dataset=True force_cache_computation=False \
  hydra/job_logging=stdout hydra.output_subdir=null 2>&1 | tee "$LOG_FILE"

#!/usr/bin/env bash
# Train ONE scene-specific privileged residual teacher from an old goal-free IL+RL expert.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need CONDA_BIN; need NAVSIM_EXP_ROOT

BUCKET_NAME="${BUCKET_NAME:-general_or_no_tag}"
BASE_RL_CKPT="${BASE_RL_CKPT:-}"
need BASE_RL_CKPT; need VLM_PATH; need NAV_CACHE; need NAV_MANIFEST; need NAV_BUCKET_ROOT

GOAL_INJECTION="${GOAL_INJECTION:-gated_cross}"
GOAL_POINT_MODE="${GOAL_POINT_MODE:-final}"
GOAL_INDICES="${GOAL_INDICES:-[1,4,7]}"
TRAIN_LAST_N_DIT_BLOCKS="${TRAIN_LAST_N_DIT_BLOCKS:-0}"
ADAPTER_LR="${ADAPTER_LR:-5e-5}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
SIM_RATIO="${SIM_RATIO:-0.40}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPUS="${GPUS:-8}"; NNODES="${NNODES:-1}"; NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-23631}"

NAV_TOKENS="${NAV_BUCKET_ROOT}/exclusive_${BUCKET_NAME}_tokens.json"
[[ -f "$NAV_TOKENS" ]] || { echo "missing $NAV_TOKENS" >&2; exit 2; }

SIM_PATHS=(); SIM_MANIFESTS=(); SIM_TOKENS=()
for r in 0 1; do
  eval cache="\${SIM_CACHE_R${r}:-}"
  eval manifest="\${SIM_MANIFEST_R${r}:-}"
  eval root="\${SIM_BUCKET_R${r}_ROOT:-}"
  tok="${root:-}/exclusive_${BUCKET_NAME}_tokens.json"
  if [[ -n "${cache:-}" && -d "$cache" && -n "${manifest:-}" && -f "$manifest" && -f "$tok" ]]; then
    SIM_PATHS+=("$cache"); SIM_MANIFESTS+=("$manifest"); SIM_TOKENS+=("$tok")
  fi
done
join(){ if [[ $# -eq 0 ]]; then echo '[]'; else local IFS=,; echo "[$*]"; fi; }
SIM_PATHS_ARG="$(join ${SIM_PATHS[@]+"${SIM_PATHS[@]}"})"
SIM_MANIFESTS_ARG="$(join ${SIM_MANIFESTS[@]+"${SIM_MANIFESTS[@]}"})"
SIM_TOKENS_ARG="$(join ${SIM_TOKENS[@]+"${SIM_TOKENS[@]}"})"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_priv_goal_v2_${BUCKET_NAME}_${GOAL_POINT_MODE}_${GOAL_INJECTION}}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/run.log}"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[stage3] bucket=$BUCKET_NAME init=$BASE_RL_CKPT"
echo "[stage3] goal=$GOAL_POINT_MODE/$GOAL_INJECTION epochs=$MAX_EPOCHS sim_ratio=$SIM_RATIO last_blocks=$TRAIN_LAST_N_DIT_BLOCKS"

"${CONDA_BIN}/torchrun" \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
  --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_privileged_goal_adapter_v2.py" \
  agent=recogdrive_agent_privileged_goal_adapter_v2 \
  "agent.checkpoint_path='$BASE_RL_CKPT'" "agent.vlm_path='$VLM_PATH'" \
  "agent.goal_injection='$GOAL_INJECTION'" "agent.goal_point_mode='$GOAL_POINT_MODE'" \
  "agent.goal_indices=$GOAL_INDICES" agent.train_last_n_dit_blocks="$TRAIN_LAST_N_DIT_BLOCKS" \
  agent.adapter_lr="$ADAPTER_LR" agent.backbone_lr_scale="$BACKBONE_LR_SCALE" agent.adapter_epochs="$MAX_EPOCHS" \
  "+priv_goal_nav_bucket_tokens='$NAV_TOKENS'" "+priv_goal_nav_manifest='$NAV_MANIFEST'" \
  "+priv_goal_sim_cache_paths=$SIM_PATHS_ARG" "+priv_goal_sim_manifests=$SIM_MANIFESTS_ARG" \
  "+priv_goal_sim_bucket_tokens=$SIM_TOKENS_ARG" +priv_goal_sim_ratio="$SIM_RATIO" \
  trainer.params.max_epochs="$MAX_EPOCHS" trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="$NNODES" trainer.params.devices="$GPUS" \
  trainer.params.strategy=ddp_find_unused_parameters_true trainer.params.check_val_every_n_epoch=1 \
  trainer.params.num_sanity_val_steps=0 dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers=8 dataloader.params.prefetch_factor=4 +dataloader.params.persistent_workers=true \
  experiment_name="$EXPERIMENT_NAME" train_test_split=navtrain cache_path="$NAV_CACHE" \
  use_cache_without_dataset=True force_cache_computation=False hydra/job_logging=stdout hydra.output_subdir=null \
  2>&1 | tee "$LOG_FILE"

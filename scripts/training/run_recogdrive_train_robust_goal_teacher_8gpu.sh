#!/usr/bin/env bash
# Train one robust privileged-goal IL teacher on a scene bucket.
# Default corruption: 70% clean GT goal / 20% noisy goal / 10% masked goal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
# Known-bad Alluxio cache shards to skip up-front (see data/epdms/bad_cache_shards_newvlm.txt).
export SCENE_ROUTER_BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${REPO_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"

BUCKET_NAME="${BUCKET_NAME:-general_or_no_tag}"
NNODES="${NNODES:-1}"; RANK="${RANK:-0}"; GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-23521}"

INIT_CKPT="${INIT_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
BUCKET_ROOT="${BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain}"
BUCKET_TOKENS="${BUCKET_TOKENS:-${BUCKET_ROOT}/exclusive_${BUCKET_NAME}_tokens.json}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/data/epdms/manifests/nav_train_newvlm.json}"

GOAL_DROPOUT_P="${GOAL_DROPOUT_P:-0.10}"
GOAL_NOISE_P="${GOAL_NOISE_P:-0.20}"
GOAL_NOISE_STD_XY="${GOAL_NOISE_STD_XY:-1.0}"
GOAL_NOISE_STD_HEADING="${GOAL_NOISE_STD_HEADING:-0.10}"
LR="${LR:-1e-4}"; MAX_EPOCHS="${MAX_EPOCHS:-200}"; BATCH_SIZE="${BATCH_SIZE:-16}"
RESUME_CKPT="${RESUME_CKPT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_teacher_${BUCKET_NAME}_il_goal_adaln_robust_newvlm}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/run.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
HYDRA_RESUME=(); [[ -n "${RESUME_CKPT}" ]] && HYDRA_RESUME+=("+ckpt_path='${RESUME_CKPT}'")

echo "[RobustGoalTeacher] bucket=${BUCKET_NAME} clean=$(python - <<PY
print(1-float('${GOAL_DROPOUT_P}')-float('${GOAL_NOISE_P}'))
PY
) noisy=${GOAL_NOISE_P} masked=${GOAL_DROPOUT_P} noise_xy=${GOAL_NOISE_STD_XY}m"

torchrun --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_robust_goal_teacher.py" \
  agent=recogdrive_agent_robust_goal_teacher \
  "agent.checkpoint_path='${INIT_CKPT}'" "agent.vlm_path='${VLM_PATH}'" agent.goal_mode='adaln' \
  agent.goal_dropout_p="${GOAL_DROPOUT_P}" agent.goal_noise_p="${GOAL_NOISE_P}" \
  agent.goal_noise_std_xy="${GOAL_NOISE_STD_XY}" agent.goal_noise_std_heading="${GOAL_NOISE_STD_HEADING}" \
  agent.lr="${LR}" agent.cache_hidden_state=True agent.grpo=False agent.dit_type='small' agent.vlm_size='small' agent.sampling_method='ddim' \
  "+goal_teacher_bucket_tokens='${BUCKET_TOKENS}'" "+goal_teacher_cache_manifest='${MANIFEST}'" \
  trainer.params.max_epochs="${MAX_EPOCHS}" trainer.params.precision=bf16-mixed trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" trainer.params.strategy=ddp_find_unused_parameters_true \
  trainer.params.check_val_every_n_epoch=1 trainer.params.num_sanity_val_steps=0 \
  dataloader.params.batch_size="${BATCH_SIZE}" dataloader.params.num_workers=8 dataloader.params.prefetch_factor=4 \
  +dataloader.params.persistent_workers=true experiment_name="${EXPERIMENT_NAME}" train_test_split=navtrain \
  cache_path="${CACHE_PATH}" use_cache_without_dataset=True force_cache_computation=False \
  ${HYDRA_RESUME[@]+"${HYDRA_RESUME[@]}"} hydra/job_logging=stdout hydra.output_subdir=null 2>&1 | tee "${LOG_FILE}"

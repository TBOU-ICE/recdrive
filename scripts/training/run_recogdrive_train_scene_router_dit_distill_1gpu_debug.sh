#!/usr/bin/env bash
# 2-GPU smoke/debug for scene-router four-teacher DiT OPD (v1).
# Small batch, 2-GPU DDP: validates wiring (routing / x0 / exopd) AND the
# sync_dist logging fix (per-rank bucket sets differ -> must not deadlock).
# Additive: does not modify any existing file.
set -euo pipefail

export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"

# ckpts aligned with run_recogdrive_train_scene_router_dit_distill_8gpu.sh (same config under test)
STUDENT_CKPT="${STUDENT_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_progress_curbside_stopgo_rl_newvlm/2026.07.24.05.09.01/lightning_logs/version_0/checkpoints/ckpt/epoch=14-step=2385.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_rule_intersection_rl_newvlm/2026.07.24.04.59.12/lightning_logs/version_0/checkpoints/ckpt/epoch=14-step=2115.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_safety_dynamics_interaction_rl_newvlm/2026.07.27.07.05.26/lightning_logs/version_0/checkpoints/epoch=14-step=1890.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/workspace/models/recdrive/v1.0.0/training_teacher_general_or_no_tag_rl_newvlm/2026.07.23.22.12.41/lightning_logs/version_0/checkpoints/epoch=11-step=2100.ckpt}"
EXOPD_REF_CKPT="${EXOPD_REF_CKPT:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
EXOPD_LAMBDA="${EXOPD_LAMBDA:-1.25}"
MATCH_TARGET="${MATCH_TARGET:-x0}"
TOKEN_TO_BUCKET_JSON="${TOKEN_TO_BUCKET_JSON:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"
VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-debug_scene_router_dit_opd_v1_2gpu}"
MASTER_PORT="${MASTER_PORT:-24571}"

echo "[scene-router-debug] GPUS=${GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MASTER_PORT=${MASTER_PORT}"
echo "[scene-router-debug] STUDENT_CKPT=${STUDENT_CKPT}"
echo "[scene-router-debug] TEACHER_SAFETY_CKPT=${TEACHER_SAFETY_CKPT}"

PYTHONUNBUFFERED=1 /workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
  --nnodes=1 --node_rank=0 --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_scene_router_dit_distill.py" \
  agent=recogdrive_agent_scene_router_dit_distill \
  "agent.checkpoint_path='${STUDENT_CKPT}'" \
  "agent.vlm_path='${VLM_PATH}'" \
  "agent.teacher_ckpt_progress_curbside_stopgo='${TEACHER_PROGRESS_CKPT}'" \
  "agent.teacher_ckpt_rule_intersection='${TEACHER_RULE_CKPT}'" \
  "agent.teacher_ckpt_safety_dynamics_interaction='${TEACHER_SAFETY_CKPT}'" \
  "agent.teacher_ckpt_general_or_no_tag='${TEACHER_GENERAL_CKPT}'" \
  "agent.token_to_bucket_json='${TOKEN_TO_BUCKET_JSON}'" \
  agent.teacher_select='scene_route' \
  "agent.match_target='${MATCH_TARGET}'" \
  "agent.exopd_ref_checkpoint='${EXOPD_REF_CKPT}'" \
  agent.exopd_lambda="${EXOPD_LAMBDA}" \
  agent.lr=1e-4 \
  agent.grpo=False \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  trainer.params.max_epochs=1 \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes=1 \
  trainer.params.devices="${GPUS}" \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  trainer.params.limit_train_batches=10 \
  trainer.params.limit_val_batches=2 \
  dataloader.params.batch_size=4 \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

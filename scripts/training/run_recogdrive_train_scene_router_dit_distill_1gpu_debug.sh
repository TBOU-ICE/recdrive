#!/usr/bin/env bash
# 1-GPU smoke/debug for scene-router four-teacher DiT OPD (v1).
# Small batch + single GPU to validate wiring (routing / x0 / exopd) end-to-end.
# Additive: does not modify any existing file.
set -euo pipefail

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/recdrive-multi-opd-v1}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# student init = new-vlm IL base (representation-consistent; old-vlm epoch=2 is wrong representation)
STUDENT_CKPT="${STUDENT_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_dit_il_fullmix_simscale_newvlm/2026.07.20.14.31.06/lightning_logs/version_0/checkpoints/epoch=199-step=312200.ckpt}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_teacher_progress_curbside_stopgo_rl_newvlm/2026.07.24.05.09.01/lightning_logs/version_0/checkpoints/ckpt/epoch=14-step=2385.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_teacher_rule_intersection_rl_newvlm/2026.07.24.04.59.12/lightning_logs/version_0/checkpoints/ckpt/epoch=14-step=2115.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_teacher_safety_dynamics_interaction_rl_newvlm/2026.07.24.09.41.09/lightning_logs/version_0/checkpoints/epoch=33-step=4284.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_teacher_general_or_no_tag_rl_newvlm/2026.07.23.22.12.41/lightning_logs/version_0/checkpoints/epoch=11-step=2100.ckpt}"
EXOPD_REF_CKPT="${EXOPD_REF_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_dit_il_fullmix_simscale_newvlm/2026.07.20.14.31.06/lightning_logs/version_0/checkpoints/epoch=199-step=312200.ckpt}"
EXOPD_LAMBDA="${EXOPD_LAMBDA:-1.25}"
MATCH_TARGET="${MATCH_TARGET:-x0}"
TOKEN_TO_BUCKET_JSON="${TOKEN_TO_BUCKET_JSON:-/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain/exclusive_token_to_bucket.json}"
VLM_PATH="${VLM_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/vlm_simscale_lora_merged}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-debug_scene_router_dit_opd_v1}"
MASTER_PORT="${MASTER_PORT:-23462}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
  --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_port="${MASTER_PORT}" \
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
  trainer.params.devices=1 \
  trainer.params.limit_train_batches=5 \
  trainer.params.limit_val_batches=2 \
  dataloader.params.batch_size=2 \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

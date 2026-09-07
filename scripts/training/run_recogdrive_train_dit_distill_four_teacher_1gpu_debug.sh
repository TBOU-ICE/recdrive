#!/usr/bin/env bash
# Single-GPU debug for fixed-weight four-teacher DiT OPD distillation.
# Example:
#   bash scripts/training/run_recogdrive_train_dit_distill_four_teacher_1gpu_debug.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/training/run_recogdrive_train_dit_distill_four_teacher_1gpu_debug.sh
set -euo pipefail

export PATH="/opt/conda/envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/mnt/volumes/ad-e2e-bd-su01/nby/recdrive-scene}"
export OPENSCENE_DATA_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23457}"
GPUS=1

STUDENT_CKPT="${STUDENT_CKPT:-/mnt/models/recdrive/v1.0.0/training_recogdrive_vlm_il/2026.05.19.18.10.54/lightning_logs/version_0/checkpoints/epoch=2-step=1995.ckpt}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-/mnt/models/recdrive/v1.0.0/training_recogdrive_bucket_progress_rl/2026.07.01.12.37.06/lightning_logs/version_0/checkpoints/epoch=9-step=9350.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-/mnt/models/recdrive/v1.0.0/training_recogdrive_bucket_rule_rl/2026.07.01.12.27.25/lightning_logs/version_0/checkpoints/epoch=9-step=10670.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-/mnt/models/recdrive/v1.0.0/training_recogdrive_bucket_safety_rl/2026.07.01.13.17.33/lightning_logs/version_0/checkpoints/epoch=9-step=11580.ckpt}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-/mnt/models/recdrive/v1.0.0/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"
CACHE_PATH="${CACHE_PATH:-/mnt/models/recdrive/v1.0.0/recogdrive_agent_cache_dir_train}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-debug_four_teacher_dit_opd_1gpu}"

echo "[four-teacher-1gpu] MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[four-teacher-1gpu] STUDENT_CKPT=${STUDENT_CKPT}"
echo "[four-teacher-1gpu] TEACHER_PROGRESS_CKPT=${TEACHER_PROGRESS_CKPT}"
echo "[four-teacher-1gpu] TEACHER_RULE_CKPT=${TEACHER_RULE_CKPT}"
echo "[four-teacher-1gpu] TEACHER_SAFETY_CKPT=${TEACHER_SAFETY_CKPT}"
echo "[four-teacher-1gpu] TEACHER_GENERAL_CKPT=${TEACHER_GENERAL_CKPT}"

/opt/conda/envs/recdrive/bin/torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_four_teacher_dit_distill.py" \
  agent=recogdrive_agent_four_teacher_dit_distill \
  "agent.checkpoint_path='${STUDENT_CKPT}'" \
  "agent.teacher_dit_checkpoint_progress='${TEACHER_PROGRESS_CKPT}'" \
  "agent.teacher_dit_checkpoint_rule='${TEACHER_RULE_CKPT}'" \
  "agent.teacher_dit_checkpoint_safety='${TEACHER_SAFETY_CKPT}'" \
  "agent.teacher_dit_checkpoint_general='${TEACHER_GENERAL_CKPT}'" \
  agent.teacher_weight_progress=0.3 \
  agent.teacher_weight_rule=0.2 \
  agent.teacher_weight_safety=0.2 \
  agent.teacher_weight_general=0.3 \
  agent.dit_distill_smooth_weight=0.02 \
  agent.dit_distill_min_sigma=0.04 \
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
  trainer.params.strategy=auto \
  trainer.params.limit_train_batches=20 \
  trainer.params.limit_val_batches=5 \
  dataloader.params.batch_size=1 \
  dataloader.params.num_workers=2 \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

#!/usr/bin/env bash
set -euo pipefail

# 1-GPU smoke / debug run for ReCogDrive DiT-OPD.
# Reduces batch_size and max_epochs for quick pipeline validation.
# Override any variable from the environment.

export PATH="${RECDRIVE_CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/recdrive-opd-vlm-dit-opd-gpt}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
CACHE_PATH="${CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/volumes/ad-e2e-al-sh01/nby/training_recogdrive_dit_opd_gpt_1gpu_debug}"

STUDENT_CKPT="${STUDENT_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_vlm_il/2026.05.19.18.10.54/lightning_logs/version_0/checkpoints/epoch=0-step=665.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"

MASTER_PORT="${MASTER_PORT:-23457}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-50}"

if ! command -v torchrun >/dev/null 2>&1; then
  echo "[ERROR] torchrun not found in PATH. Check RECDRIVE_CONDA_BIN/PATH."
  exit 1
fi

if [ ! -e "$STUDENT_CKPT" ]; then
  echo "[ERROR] STUDENT_CKPT does not exist: $STUDENT_CKPT"
  exit 1
fi
if [ ! -e "$TEACHER_CKPT" ]; then
  echo "[ERROR] TEACHER_CKPT does not exist: $TEACHER_CKPT"
  exit 1
fi
if [ ! -d "$CACHE_PATH" ]; then
  echo "[ERROR] CACHE_PATH does not exist or is not a directory: $CACHE_PATH"
  exit 1
fi

echo "[1gpu-debug] DiT-OPD launch"
echo "  NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  CACHE_PATH=$CACHE_PATH"
echo "  STUDENT_CKPT=$STUDENT_CKPT"
echo "  TEACHER_CKPT=$TEACHER_CKPT"
echo "  VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=1 \
  --master_port=${MASTER_PORT} \
  ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_rl.py \
  agent=recogdrive_agent_dit_opd \
  "agent.checkpoint_path='${STUDENT_CKPT}'" \
  "agent.teacher_dit_checkpoint='${TEACHER_CKPT}'" \
  agent.lr=1e-5 \
  agent.dit_opd=True \
  agent.dit_distill=False \
  agent.opd=False \
  agent.grpo=False \
  agent.cache_hidden_state=True \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  "output_dir='${OUTPUT_DIR}'" \
  trainer.params.max_epochs=3 \
  trainer.params.num_nodes=1 \
  trainer.params.devices=1 \
  trainer.params.precision=16-mixed \
  trainer.params.val_check_interval=${VAL_CHECK_INTERVAL} \
  dataloader.params.batch_size=4 \
  experiment_name=training_recogdrive_dit_opd_gpt_1gpu_debug \
  "train_test_split=${TRAIN_TEST_SPLIT}" \
  "cache_path='${CACHE_PATH}'" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

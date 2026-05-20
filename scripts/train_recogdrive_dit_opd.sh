#!/usr/bin/env bash
set -euo pipefail

# Pure ReCogDrive DiT-OPD launch script.
# This follows the existing ReCogDrive distributed entrypoint. Override paths
# from the environment when your machine uses a different mount layout.

export PATH="${RECDRIVE_CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/code}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
CACHE_PATH="${CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/volumes/ad-e2e-al-sh01/nby/training_recogdrive_dit_opd_gpt_v1}"
DIT_OPD_DEBUG_LOG_INTERVAL="${DIT_OPD_DEBUG_LOG_INTERVAL:-50}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"

STUDENT_CKPT="${STUDENT_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_vlm_il/2026.05.19.18.10.54/lightning_logs/version_0/checkpoints/epoch=2-step=1995.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS="${GPUS:-8}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

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

echo "DiT-OPD launch"
echo "  NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT"
echo "  CACHE_PATH=$CACHE_PATH"
echo "  STUDENT_CKPT=$STUDENT_CKPT"
echo "  TEACHER_CKPT=$TEACHER_CKPT"
echo "  GPUS=$GPUS NNODES=$NNODES RANK=$RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "  DIT_OPD_DEBUG_LOG_INTERVAL=$DIT_OPD_DEBUG_LOG_INTERVAL"
echo "  VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL"

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${GPUS} \
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
  trainer.params.max_epochs=200 \
  trainer.params.num_nodes=${NNODES} \
  trainer.params.devices=${GPUS} \
  trainer.params.precision=16-mixed \
  trainer.params.val_check_interval=${VAL_CHECK_INTERVAL} \
  dataloader.params.batch_size=16 \
  experiment_name=training_recogdrive_dit_opd_gpt_v1 \
  "train_test_split=${TRAIN_TEST_SPLIT}" \
  "cache_path='${CACHE_PATH}'" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  +dit_opd_debug_log_interval=${DIT_OPD_DEBUG_LOG_INTERVAL} \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

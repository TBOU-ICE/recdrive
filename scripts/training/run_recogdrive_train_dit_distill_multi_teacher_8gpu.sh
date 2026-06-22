#!/usr/bin/env bash
# 8-GPU fixed-weight dual-teacher DiT OPD training (2B)
# student DiT: trainable, initialized from IL checkpoint (epoch=2 for better logvar calibration)
# teacher IL DiT: frozen, EC-oriented
# teacher RL DiT: frozen, PDMS-oriented
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
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS="${GPUS:-8}"

# epoch=2 checkpoint: more IL training → better logvar calibration → stable distillation start
CHECKPOINT="${CHECKPOINT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_vlm_il/2026.05.19.18.10.54/lightning_logs/version_0/checkpoints/epoch=2-step=1995.ckpt}"
TEACHER_IL_CKPT="${TEACHER_IL_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_temporal_il_opd_2b/2026.06.04.00.02.53/checkpoints/temporal-epochepoch=023-stepstep=00053000.ckpt}"
TEACHER_RL_CKPT="${TEACHER_RL_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_mini_difference_comparation2-v2}"
CACHE_PATH="${CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
LOG_ROOT="${LOG_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/training_mini_difference_comparation2-v2}"

echo "[8gpu] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[8gpu] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "[8gpu] CHECKPOINT=${CHECKPOINT}"
echo "[8gpu] TEACHER_IL_CKPT=${TEACHER_IL_CKPT}"
echo "[8gpu] TEACHER_RL_CKPT=${TEACHER_RL_CKPT}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_rl.py" \
  agent=recogdrive_agent_dit_distill \
  agent.lr=1e-4 \
  agent.dit_distill=True \
  agent.cache_hidden_state=True \
  "agent.checkpoint_path='${CHECKPOINT}'" \
  "agent.teacher_dit_checkpoint_il='${TEACHER_IL_CKPT}'" \
  "agent.teacher_dit_checkpoint_rl='${TEACHER_RL_CKPT}'" \
  agent.dit_distill_il_weight=0.75 \
  agent.dit_distill_rl_weight=0.25 \
  agent.dit_distill_smooth_weight=0.02 \
  agent.dit_distill_eps_clip=0.2 \
  agent.dit_distill_min_sigma=0.04 \
  agent.dit_distill_normalize_advantage=True \
  "agent.dit_distill_log_dir='${LOG_ROOT}'" \
  agent.dit_distill_log_interval=50 \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  agent.grpo=False \
  agent.opd=False \
  trainer.params.max_epochs=50 \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  dataloader.params.batch_size=8 \
  logger.type=tensorboard \
  "+logger.save_dir='${LOG_ROOT}/tensorboard'" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

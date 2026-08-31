#!/usr/bin/env bash
# 8-GPU pure OPRD multi-teacher DiT representation distillation training.
# Additive script: no original training file is modified.
set -euo pipefail

export PATH="/opt/conda/envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/mnt/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1}"
export OPENSCENE_DATA_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download"
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
MASTER_PORT="${MASTER_PORT:-23459}"
GPUS="${GPUS:-8}"

CHECKPOINT="${CHECKPOINT:-/mnt/models/recdrive/v1.0.0/training_recogdrive_vlm_il/2026.05.19.18.10.54/lightning_logs/version_0/checkpoints/epoch=2-step=1995.ckpt}"
TEACHER_IL_CKPT="${TEACHER_IL_CKPT:-/mnt/models/recdrive/v1.0.0/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}"
TEACHER_RL_CKPT="${TEACHER_RL_CKPT:-/mnt/models/recdrive/v1.0.0/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"
CACHE_PATH="${CACHE_PATH:-/mnt/models/recdrive/v1.0.0/recogdrive_agent_cache_dir_train}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_oprd_multi_teacher_dit_2b_v2}"
LOG_ROOT="${LOG_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/training_oprd_multi_teacher_dit_2b_v2}"

# Pure OPRD weights.
OPRD_MID_IL_WEIGHT="${OPRD_MID_IL_WEIGHT:-1.0}"
OPRD_LAST_IL_WEIGHT="${OPRD_LAST_IL_WEIGHT:-0.85}"
OPRD_LAST_RL_WEIGHT="${OPRD_LAST_RL_WEIGHT:-0.15}"
OPRD_FINAL_REPR_WEIGHT="${OPRD_FINAL_REPR_WEIGHT:-1.0}"
# Trajectory-level loss weights (fix action_decoder gradient).
# Mirrors the IL/RL split used for hidden-state losses.
OPRD_TRAJ_IL_WEIGHT="${OPRD_TRAJ_IL_WEIGHT:-0.85}"
OPRD_TRAJ_RL_WEIGHT="${OPRD_TRAJ_RL_WEIGHT:-0.15}"

PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
TEMPORAL_MIN_DT_S="${TEMPORAL_MIN_DT_S:-0.1}"
TEMPORAL_MAX_DT_S="${TEMPORAL_MAX_DT_S:-1.1}"

# Reuse the temporal pair dataloader/run file. The agent's trainer uses
# hidden-state OPRD losses plus an optional trajectory-level MSE loss
# (OPRD_TRAJ_IL/RL_WEIGHT) that gives action_decoder a gradient path.
echo "[oprd-mt-dit] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[oprd-mt-dit] CHECKPOINT=${CHECKPOINT}"
echo "[oprd-mt-dit] TEACHER_IL_CKPT=${TEACHER_IL_CKPT}"
echo "[oprd-mt-dit] TEACHER_RL_CKPT=${TEACHER_RL_CKPT}"
echo "[oprd-mt-dit] weights: MID_IL=${OPRD_MID_IL_WEIGHT} LAST_IL=${OPRD_LAST_IL_WEIGHT} LAST_RL=${OPRD_LAST_RL_WEIGHT} FINAL=${OPRD_FINAL_REPR_WEIGHT} TRAJ_IL=${OPRD_TRAJ_IL_WEIGHT} TRAJ_RL=${OPRD_TRAJ_RL_WEIGHT}"
echo "[oprd-mt-dit] PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE} (real batch = $((PAIR_BATCH_SIZE * 2)))"

/opt/conda/envs/recdrive/bin/torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_temporal_multi_teacher_dit_distill.py" \
  agent=recogdrive_agent_oprd_multi_teacher_dit \
  agent.lr=1e-4 \
  agent.dit_distill=True \
  agent.cache_hidden_state=True \
  "agent.checkpoint_path='${CHECKPOINT}'" \
  "agent.teacher_dit_checkpoint_il='${TEACHER_IL_CKPT}'" \
  "agent.teacher_dit_checkpoint_rl='${TEACHER_RL_CKPT}'" \
  agent.dit_distill_eps_clip=0.2 \
  agent.dit_distill_min_sigma=0.04 \
  agent.dit_distill_normalize_advantage=True \
  "agent.dit_distill_log_dir='${LOG_ROOT}'" \
  agent.dit_distill_log_interval=50 \
  agent.oprd_mid_il_weight="${OPRD_MID_IL_WEIGHT}" \
  agent.oprd_last_il_weight="${OPRD_LAST_IL_WEIGHT}" \
  agent.oprd_last_rl_weight="${OPRD_LAST_RL_WEIGHT}" \
  agent.oprd_use_final_repr=True \
  agent.oprd_final_repr_weight="${OPRD_FINAL_REPR_WEIGHT}" \
  agent.oprd_traj_il_weight="${OPRD_TRAJ_IL_WEIGHT}" \
  agent.oprd_traj_rl_weight="${OPRD_TRAJ_RL_WEIGHT}" \
  agent.oprd_normalize_hidden=True \
  agent.oprd_loss_type='mse' \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  agent.grpo=False \
  agent.opd=False \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  dataloader.params.batch_size="${PAIR_BATCH_SIZE}" \
  logger.type=tensorboard \
  "+logger.save_dir='${LOG_ROOT}/tensorboard'" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=False \
  force_cache_computation=False \
  +temporal_min_dt_s="${TEMPORAL_MIN_DT_S}" \
  +temporal_max_dt_s="${TEMPORAL_MAX_DT_S}" \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

#!/usr/bin/env bash
# Single-GPU debug for fixed-weight dual-teacher DiT OPD training (2B).
# Same recipe as run_recogdrive_train_dit_distill_multi_teacher_8gpu.sh, one process.
# student DiT: trainable, initialized from IL checkpoint (epoch=2 for better logvar calibration)
# teacher IL DiT: frozen, EC-oriented
# teacher RL DiT: frozen, PDMS-oriented
#
# Usage:
#   bash run_recogdrive_train_dit_distill_multi_teacher_1gpu.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_recogdrive_train_dit_distill_multi_teacher_1gpu.sh

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
_TRAIN_PY="navsim/planning/script/run_training_recogdrive_rl.py"
_CLUSTER_DEVKIT="/mnt/volumes/ad-e2e-bd-su01/nby/recdrive"

if [ -n "${NAVSIM_DEVKIT_ROOT:-}" ]; then
  export NAVSIM_DEVKIT_ROOT
elif [ -f "$_CLUSTER_DEVKIT/$_TRAIN_PY" ]; then
  export NAVSIM_DEVKIT_ROOT="$_CLUSTER_DEVKIT"
else
  export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
fi

if [ ! -f "$NAVSIM_DEVKIT_ROOT/$_TRAIN_PY" ]; then
  echo "error: missing $_TRAIN_PY under NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT" >&2
  exit 2
fi

export PATH="/opt/conda/envs/recdrive/bin:${PATH:-}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-bd-su01/nby/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23669}"
export MASTER_PORT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPUS=1
NNODES=1
RANK=0

# epoch=2 checkpoint: more IL training → better logvar calibration → stable distillation start
CHECKPOINT="${CHECKPOINT:-/mnt/models/recdrive/v1.0.0/training_recogdrive_vlm_il/2026.05.19.18.10.54/lightning_logs/version_0/checkpoints/epoch=2-step=1995.ckpt}"
TEACHER_IL_CKPT="${TEACHER_IL_CKPT:-/mnt/models/recdrive/v1.0.0/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}"
TEACHER_RL_CKPT="${TEACHER_RL_CKPT:-/mnt/models/recdrive/v1.0.0/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_dual_teacher_dit_opd_2b_1gpu_debug}"
CACHE_PATH="${CACHE_PATH:-/mnt/models/recdrive/v1.0.0/recogdrive_agent_cache_dir_train}"
LOG_ROOT="${LOG_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/training_dual_teacher_dit_opd_2b_1gpu_debug}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

echo "[1gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[1gpu] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "[1gpu] CHECKPOINT=${CHECKPOINT}"
echo "[1gpu] TEACHER_IL_CKPT=${TEACHER_IL_CKPT}"
echo "[1gpu] TEACHER_RL_CKPT=${TEACHER_RL_CKPT}"
echo "[1gpu] LOG_ROOT=${LOG_ROOT}"

_TORCHRUN_NBY="/opt/conda/envs/recdrive/bin/torchrun"
_TORCHRUN_VOL="/mnt/volumes/nby/conda_envs/recdrive/bin/torchrun"
if [ -x "$_TORCHRUN_NBY" ]; then
  TORCHRUN="$_TORCHRUN_NBY"
elif [ -x "$_TORCHRUN_VOL" ]; then
  TORCHRUN="$_TORCHRUN_VOL"
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN="torchrun"
else
  echo "error: torchrun not found (tried $_TORCHRUN_NBY, $_TORCHRUN_VOL, PATH)" >&2
  exit 2
fi

"$TORCHRUN" \
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
  trainer.params.max_epochs=3 \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  dataloader.params.batch_size=2 \
  dataloader.params.num_workers=2 \
  logger.type=tensorboard \
  "+logger.save_dir='${LOG_ROOT}/tensorboard'" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

#!/bin/sh
# Single-GPU debug for DiT OPD (On-Policy Distillation) training (ReCogDrive).
# Optional: CUDA_VISIBLE_DEVICES=0 sh run_recogdrive_train_dit_distill_1gpu_debug.sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
_TRAIN_PY="navsim/planning/script/run_training_recogdrive_rl.py"
_CLUSTER_DEVKIT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive"

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

export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-23669}
export MASTER_PORT
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NPROC_PER_NODE=1

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "1-GPU DiT-distill debug: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

_TORCHRUN_NBY="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun"
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
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py" \
  agent=recogdrive_agent_dit_distill \
  agent.dit_distill=True \
  agent.cache_hidden_state=True \
  agent.lr=5e-5 \
  "agent.checkpoint_path='/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_vlm_il/2026.05.19.18.10.54/lightning_logs/version_0/checkpoints/epoch=2-step=1995.ckpt'" \
  agent.teacher_dit_checkpoint='/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt' \
  agent.dit_distill_eps_clip=0.2 \
  agent.dit_distill_min_sigma=0.04 \
  agent.dit_distill_normalize_advantage=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  agent.grpo=False \
  agent.opd=False \
  trainer.params.max_epochs=3 \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes=1 \
  trainer.params.devices=1 \
  dataloader.params.batch_size=2 \
  dataloader.params.num_workers=2 \
  logger.type=tensorboard \
  experiment_name=training_recogdrive_dit_opd_1gpu_debug \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

#!/bin/sh
# Single-GPU debug: same training recipe as run_recogdrive_train_multi_node_2b.sh, one process.
# POSIX sh (dash): `sh run_recogdrive_train_1gpu_debug.sh`
# Optional: CUDA_VISIBLE_DEVICES=0 sh run_recogdrive_train_1gpu_debug.sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
_TRAIN_PY="navsim/planning/script/run_training_recogdrive.py"
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

export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-bd-su01/nby/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"

TRAIN_TEST_SPLIT=navtrain
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
export MASTER_PORT
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NPROC_PER_NODE=1

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "1-GPU debug: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
export CUDA_LAUNCH_BLOCKING=1

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
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive.py" \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path='/mnt/models/recdrive/v1.0.0/ReCogDrive-VLM-2B' \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=200 \
    trainer.params.num_nodes=1 \
    trainer.params.devices=1 \
    experiment_name=training_recogdrive_vlm_nby_1gpu_debug \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/recogdrive_agent_cache_dir_train_jiaoqf" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    hydra/job_logging=stdout \
    hydra.output_subdir=null

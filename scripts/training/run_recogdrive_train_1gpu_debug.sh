#!/bin/sh
# Single-GPU debug entrypoint (no MLP_* / multi-node env required).
# POSIX sh (dash): use `sh run_recogdrive_train_1gpu_debug.sh` — no pipefail (not in POSIX).
# Optional: CUDA_VISIBLE_DEVICES=3 sh run_recogdrive_train_1gpu_debug.sh
#
# NAVSIM_DEVKIT_ROOT / torchrun: cluster defaults below are used only when present;
# otherwise this checkout (parent of navsim/) and PATH are used so local/CI runs work.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
_TRAIN_PY="navsim/planning/script/run_training_recogdrive.py"
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

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"

TRAIN_TEST_SPLIT=navtrain
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
export MASTER_PORT
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# Pin to one visible GPU unless user already set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NPROC_PER_NODE=1

echo "1-GPU debug: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
export CUDA_LAUNCH_BLOCKING=1

_CONDA_TORCHRUN="/mnt/volumes/nby/conda_envs/recdrive/bin/torchrun"
if [ -x "$_CONDA_TORCHRUN" ]; then
  TORCHRUN="$_CONDA_TORCHRUN"
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN="torchrun"
else
  echo "error: torchrun not found (tried $_CONDA_TORCHRUN and PATH)" >&2
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
    agent.vlm_path='/path/to/pretrain_model' \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=200 \
    trainer.params.num_nodes=1 \
    trainer.params.devices=1 \
    experiment_name=training_recogdrive_agent_1gpu_debug \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/path/to/recogdrive_agent_cache_dir_train_2b" \
    use_cache_without_dataset=True \
    # force_cache_computation=False > train_recogdrive_exp_2b_1gpu_debug.txt 2>&1

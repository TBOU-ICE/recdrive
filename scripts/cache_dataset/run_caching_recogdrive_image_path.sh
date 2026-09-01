#!/usr/bin/env bash
# Cache lightweight features for OPD training:
# stores image_path_tensor, history_trajectory, high_command_one_hot, status_feature
# (no VLM forward pass, no hidden states)
# This cache is used by OPD training where VLM runs online.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$REPO_ROOT}"

TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/exp"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"

# Separate cache dir from the hidden-state cache
CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_opd

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-63670}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPUS:-8}}"

export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

echo "Caching image-path features for OPD training..."
echo "Cache path: $CACHE_PATH"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"

/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=${GPUS_PER_NODE} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache_opd \
    agent.cam_type='single' \
    agent.cache_hidden_state=False \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="" \
    cache_path=$CACHE_PATH \
    worker=sequential \
    > /mnt/volumes/ad-e2e-al-sh01/nby/log/caching_opd_rank${NODE_RANK}.txt 2>&1

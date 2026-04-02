TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp"

# export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/volumes/ad-e2e-al-sh01/cy/recdrive"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_jiaoqf
#CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_nby

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"

torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B" \
    cache_path=$CACHE_PATH \
    worker=sequential
    #> /mnt/volumes/ad-e2e-al-sh01/cy/log/caching_dataset.txt 2>&1
export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH" #nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/code"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
TRAIN_TEST_SPLIT=navtrain
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is empty}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS="${GPUS:-8}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

PORT=${PORT:-63665}

echo "GPUS: ${GPUS}"
echo "NNODES: ${NNODES}"
echo "RANK: ${RANK}"
export CUDA_LAUNCH_BLOCKING=0



/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path='/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B' \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=200 \
    trainer.params.num_nodes=${NNODES} \
    trainer.params.devices=${GPUS} \
    experiment_name=training_recogdrive_vlm_nby \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    hydra/job_logging=stdout \
    hydra.output_subdir=null
    # > /mnt/volumes/ad-e2e-al-sh01/cy/log/train_recogdrive_exp_2b.txt 2>&1

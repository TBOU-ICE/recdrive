export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH" #nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"


TRAIN_TEST_SPLIT=navtrain
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  #nby

export PYTHONPATH="/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1:${PYTHONPATH:-}"
# MASTER_PORT=${MASTER_PORT:-63669}
# PORT=${PORT:-63665}
# GPUS=${GPUS:-8}
# GPUS_PER_NODE=${GPUS_PER_NODE:-8}
# NODES=$((GPUS / GPUS_PER_NODE))
# export MASTER_PORT=${MASTER_PORT}
# export PORT=${PORT}

# echo "GPUS: ${GPUS}"
# export CUDA_LAUNCH_BLOCKING=1
NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is empty}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS=${GPUS:-8}
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

PORT=${PORT:-63665}

echo "GPUS per Node: ${GPUS}"
echo "Total Nodes (NNODES): ${NNODES}"
echo "Current Node Rank: ${RANK}"
echo "Master Addr: ${MASTER_ADDR}"

export CUDA_LAUNCH_BLOCKING=0

#CHECKPOINT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/training_recogdrive_agent/2026.03.03.07.42.42/lightning_logs/version_0/checkpoints/epoch-196_step-32899.ckpt"
CHECKPOINT="/workspace/models/recdrive/v1.0.0/training_recogdrive_vlm_nby/2026.04.04.00.28.46/lightning_logs/version_0/checkpoints/epoch=199-step=133000.ckpt"



/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.vlm_path='/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/outs' \
    agent.cam_type='single' \
    agent.grpo=True \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    agent.metric_cache_path="/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_train" \
    agent.reference_policy_checkpoint="'$CHECKPOINT'" \
    trainer.params.max_epochs=10 \
    trainer.params.num_nodes=${NNODES} \
    trainer.params.devices=${GPUS} \
    dataloader.params.batch_size=8 \
    experiment_name=training_recogdrive_agent_rl \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/workspace/models/recdrive/v1.0.0/recogdrive_agent_cache_dir_train" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    hydra/job_logging=stdout \
    hydra.output_subdir=null
    # > train_recogdrive_rl_2b.txt 2>&1
  # 2>&1 | tee -a "training_log.txt" &


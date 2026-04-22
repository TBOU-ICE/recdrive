#!/usr/bin/env bash
# DiT distillation training:
#   student DiT  ← 2B IL checkpoint (trainable)
#   teacher DiT  ← 2B RL checkpoint (frozen)
#   VLM          ← not loaded; uses pre-cached hidden states
#
# Loss = w(PDM_score) * MSE(student_pred_noise, teacher_pred_noise)

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/code}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

TRAIN_TEST_SPLIT=navtrain
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS="${GPUS:-8}"

echo "GPUS: ${GPUS}  NNODES: ${NNODES}  RANK: ${RANK}  MASTER_ADDR: ${MASTER_ADDR}  NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent_dit_distill \
    agent.lr=1e-4 \
    agent.dit_distill=True \
    agent.cache_hidden_state=True \
    agent.checkpoint_path='/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt' \
    agent.teacher_dit_checkpoint='/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt' \
    agent.metric_cache_path='/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train' \
    agent.dit_distill_reward_weight_mode='normalize' \
    agent.dit_distill_use_reward_weighting=True \
    agent.vlm_type='internvl' \
    agent.dit_type='small' \
    agent.vlm_size='small' \
    agent.sampling_method='ddim' \
    agent.grpo=False \
    agent.opd=False \
    trainer.params.max_epochs=10 \
    trainer.params.precision=bf16-true \
    trainer.params.num_nodes=${NNODES} \
    trainer.params.devices=${GPUS} \
    dataloader.params.batch_size=8 \
    experiment_name=training_recogdrive_dit_distill_2b \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    hydra/job_logging=stdout \
    hydra.output_subdir=null

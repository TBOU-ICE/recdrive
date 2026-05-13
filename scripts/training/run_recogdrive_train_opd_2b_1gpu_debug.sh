#!/usr/bin/env bash
# Single-GPU smoke / debug for OPD training (same Hydra overrides as 8-card script, lighter).
#
# Usage:
#   bash scripts/training/run_recogdrive_train_opd_2b_1gpu_debug.sh
# Optional overrides:
#   NAVSIM_DEVKIT_ROOT=/path/to/repo MASTER_ADDR=127.0.0.1 bash scripts/training/...
#
# After this runs, scale up with run_recogdrive_train_opd_2b.sh (set GPUS=8, restore limits).

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/recdrive-opd-vlm}"
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
GPUS=1

echo "[1-GPU DEBUG] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}"

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_rl.py" \
    agent=recogdrive_agent_opd \
    agent.lr=2e-6 \
    agent.opd=True \
    agent.cache_hidden_state=False \
    agent.vlm_path='/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B' \
    agent.teacher_vlm_path='/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-8B' \
    agent.checkpoint_path='/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt' \
    agent.opd_topk=32 \
    agent.opd_group_size=6 \
    agent.opd_max_new_tokens=256 \
    agent.vlm_type='internvl' \
    agent.dit_type='small' \
    agent.vlm_size='small' \
    agent.sampling_method='ddim' \
    agent.grpo=False \
    trainer.params.max_epochs=1 \
    trainer.params.precision=bf16-true \
    trainer.params.num_nodes="${NNODES}" \
    trainer.params.devices="${GPUS}" \
    trainer.params.limit_train_batches=4 \
    trainer.params.limit_val_batches=2 \
    dataloader.params.batch_size=1 \
    dataloader.params.num_workers=2 \
    experiment_name=training_recogdrive_8b_teacher_opd_2b_1gpu_debug \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_opd" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    hydra/job_logging=stdout \
    hydra.output_subdir=null

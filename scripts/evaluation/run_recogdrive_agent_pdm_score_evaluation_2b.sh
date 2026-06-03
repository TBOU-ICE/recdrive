set -x

TRAIN_TEST_SPLIT=navtest
export OPENBLAS_CORETYPE=Haswell
export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH" #nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/code"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"


export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"


#CHECKPOINT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/training_recogdrive_agent_rl_jiaoqf/2026.03.07.11.46.20/lightning_logs/version_0/checkpoints/epoch=9-step=3330.ckpt"
#CHECKPOINT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"
# PDMS on navtest: use metric_cache (eval); metric_cache_train is for training and won't match navtest tokens.
METRIC_CACHE_PATH="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache"

# 1. Set NAVSIM dataset and related environment variables
# 2. Configure torchrun (e.g., single machine: --nproc_per_node=8; adjust for multi-node)
# 3. Set agent.vlm_path and agent.checkpoint_path CHECKPOINT


/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
    --nproc_per_node=8 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path='/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/merged_model/InternVL3-2B-ckpt400-merged' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name=InternVL3-400step-opd-step2000 \
    worker=sequential

set -x

# 8-GPU PDMS eval for run_pdm_score_recogdrive (2B agent), rule_intersection bucket.
# - torchrun --nproc_per_node=8: shard scenarios across 8 GPUs via InferenceSampler.
# - worker=sequential: each rank runs inference locally (no Ray cluster).
# - PYTHONUNBUFFERED: logs flush immediately.

TRAIN_TEST_SPLIT=navtest_rule_intersection
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH" #nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/eval_recdrive_v2"
export NAVSIM_DEVKIT_ROOT="/workspace/code"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-62665}
PORT=${PORT:-62664}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GPUS=${GPUS}, GPUS_PER_NODE=${GPUS_PER_NODE}"

CHECKPOINT="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_bucket_rule_rl/2026.07.01.12.27.25/lightning_logs/version_0/checkpoints/epoch=9-step=10670.ckpt"

# PDMS on navtest must use caches built for that split; metric_cache_train tokens won't match navtest.
METRIC_CACHE_PATH="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/exp/metric_cache"

# Hydra treats "=" in override values as syntax; Lightning ckpt names must be quoted.
/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    "agent.checkpoint_path=\"${CHECKPOINT}\"" \
    agent.vlm_path='/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name=eval-pdms-rule_intersection-rl-epoch9 \
    worker=sequential

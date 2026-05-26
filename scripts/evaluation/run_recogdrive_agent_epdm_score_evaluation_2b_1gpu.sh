set -x

# Single-GPU EPDMS evaluation for run_epdm_score_recogdrive (2B agent).
# - worker=sequential: no local Ray cluster (faster startup; matches single-process eval).
# - PYTHONUNBUFFERED: stage timing logs appear immediately in terminal + log.txt.

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENBLAS_CORETYPE=Haswell

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH" # nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/recdrive"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63670}
PORT=${PORT:-63667}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# DiT / diffusion planner checkpoint
CHECKPOINT="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_dit_opd_1gpu_debug-c3/2026.05.23.17.15.25/checkpoints/epochepoch=000-stepstep=00001000.ckpt"
# Merged VLM: InternVL3-2B base + finetuned weights
VLM_PATH="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B"
VLM_WEIGHTS_PATH=""
# EPDMS on navtest uses the same metric_cache as PDMS.
METRIC_CACHE_PATH="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/exp/metric_cache"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_epdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path="'$VLM_PATH'" \
    agent.vlm_weights_path="'$VLM_WEIGHTS_PATH'" \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name=eval_recogdrive_hydramdpp_epdms_1gpu \
    worker=sequential 
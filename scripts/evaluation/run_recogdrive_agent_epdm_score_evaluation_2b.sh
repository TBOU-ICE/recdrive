set -x

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}
export OPENBLAS_CORETYPE=Haswell
export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH" # nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-scene"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"

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

# DiT / diffusion planner checkpoint
CHECKPOINT="/workspace/models/recdrive/v1.0.0/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt"
# Merged VLM: InternVL3-2B base + finetuned weights
VLM_PATH="/workspace/models/recdrive/v1.0.0/ReCogDrive-VLM-2B"
VLM_WEIGHTS_PATH=""
# EPDMS on navtest uses the same metric_cache as PDMS.
METRIC_CACHE_PATH="/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache"

/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_epdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
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
    experiment_name=eval_recogdrive_hydramdpp_epdms_il \
    worker=sequential

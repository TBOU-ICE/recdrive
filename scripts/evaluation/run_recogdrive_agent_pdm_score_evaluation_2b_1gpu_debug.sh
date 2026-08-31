set -x

# Single-GPU smoke / debug for run_pdm_score_recogdrive (2B agent).
# - train_test_split.scene_filter.max_scenes=8: quick pipeline check (remove for full navtest).
# - worker=sequential: no local Ray cluster (faster startup; matches single-process eval).
# - PYTHONUNBUFFERED: stage timing logs appear immediately in terminal + log.txt.

TRAIN_TEST_SPLIT=navtest
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PATH="/opt/conda/envs/recdrive/bin:$PATH" #nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/volumes/ad-e2e-bd-su01/nby/recdrive-multi-opd-v1-opd-dit"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

#CHECKPOINT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/training_recogdrive_agent_rl_jiaoqf/2026.03.07.11.46.20/lightning_logs/version_0/checkpoints/epoch=9-step=3330.ckpt"
#CHECKPOINT="/mnt/models/recdrive/v1.0.0/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"

# PDMS on navtest must use caches built for that split; metric_cache_train tokens won't match navtest.
METRIC_CACHE_PATH="/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache"

# Hydra treats "=" in override values as syntax; Lightning names like epoch=4-step=6650.ckpt must be quoted.
/opt/conda/envs/recdrive/bin/torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    "agent.checkpoint_path=\"${CHECKPOINT}\"" \
    agent.vlm_path='/mnt/models/recdrive/v1.0.0/merged_model/InternVL3-2B-ckpt400-merged' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name=InternVL3-400step-opd-step5000 \
    worker=sequential
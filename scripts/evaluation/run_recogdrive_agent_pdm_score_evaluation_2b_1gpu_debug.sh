set -x

# Single-GPU smoke test for run_pdm_score_recogdrive (2B agent).
# - Uses torchrun with one process so the same entrypoint / NCCL path as multi-GPU runs.
# - worker=sequential avoids starting Ray per rank (safe when WORLD_SIZE=1).
# - Default: only max_scenes scenarios for a quick pipeline check. Comment out the hydra
#   line train_test_split.scene_filter.max_scenes=... for full navtest (very slow on 1 GPU).

TRAIN_TEST_SPLIT=navtest
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH" #nby
export PYTHONPATH="/workspace/code:${PYTHONPATH:-}"  #nby

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/recdrive"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

#CHECKPOINT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/training_recogdrive_agent_rl_jiaoqf/2026.03.07.11.46.20/lightning_logs/version_0/checkpoints/epoch=9-step=3330.ckpt"
CHECKPOINT="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_agent_rl/2026.04.05.03.57.42/lightning_logs/version_0/checkpoints/epoch=9-step=13300.ckpt"

# PDMS on navtest must use caches built for that split; metric_cache_train tokens won't match navtest.
METRIC_CACHE_PATH="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache"

TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun}"

"${TORCHRUN_BIN}" \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path='/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name=recogdrive_agent_eval_1gpu_debug \
    worker=sequential \
    train_test_split.scene_filter.max_scenes=8

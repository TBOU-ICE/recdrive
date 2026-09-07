set -x

# Single-GPU smoke / debug for run_pdm_score_recogdrive (2B agent).
# - train_test_split.scene_filter.max_scenes=8: quick pipeline check (remove for full navtest).
# - worker=sequential: no local Ray cluster (faster startup; matches single-process eval).
# - PYTHONUNBUFFERED: stage timing logs appear immediately in terminal + log.txt.

TRAIN_TEST_SPLIT=navtest_rule_intersection
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

export PATH="/opt/conda/envs/recdrive/bin:$PATH" #nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/volumes/ad-e2e-bd-su01/nby/recdrive-scene"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/mnt/datasets/recdrive/20260513/nby/recdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-62667}
PORT=${PORT:-62666}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CHECKPOINT="/mnt/models/recdrive/v1.0.0/training_teacher_rule_intersection_il_goal_cross_newvlm/2026.08.02.16.45.51/lightning_logs/version_0/checkpoints/epoch=197-step=62172.ckpt"
# PDMS on navtest must use caches built for that split; metric_cache_train tokens won't match navtest.
METRIC_CACHE_PATH="/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache"

/opt/conda/envs/recdrive/bin/torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path='/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name=eval-pdms-teacher-rule-il-197epoch-rule-scene-cross \
    worker=sequential

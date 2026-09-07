set -x

# Single-GPU smoke / debug for run_pdm_score_recogdrive (2B agent).
# - train_test_split.scene_filter.max_scenes=8: quick pipeline check (remove for full navtest).
# - worker=sequential: no local Ray cluster (faster startup; matches single-process eval).
# - PYTHONUNBUFFERED: stage timing logs appear immediately in terminal + log.txt.

TRAIN_TEST_SPLIT=navtest_general_or_no_tag
#navtest_rule_intersection navtest_safety_dynamics_interaction navtest_progress_curbside_stopgo navtest_general_or_no_tag
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

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

MASTER_PORT=${MASTER_PORT:-62655}
PORT=${PORT:-62654}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CHECKPOINT="${CHECKPOINT:-/mnt/volumes/ad-e2e-bd-su01/nby/exp/training_teacher_general_or_no_tag_200il_30goal_adaln_newvlm/2026.09.02.16.54.07/lightning_logs/version_0/checkpoints/epoch=29-step=17700.ckpt}"
# GOAL_MODE must match the mode the checkpoint was TRAINED with (adaln / channel /
# cross).  A mismatch does not crash -- the checkpoint is loaded with strict=False,
# the mode-specific projection weights are silently dropped, and the goal encoder
# feeds a conditioning pathway it was never trained for -- but it wrecks the score
# (cross ckpt scored 0.46 under goal_mode=adaln vs its true goal-conditioned score).
GOAL_MODE="${GOAL_MODE:-adaln}"
# PDMS on navtest must use caches built for that split; metric_cache_train tokens won't match navtest.
METRIC_CACHE_PATH="/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache"

/opt/conda/envs/recdrive/bin/torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive_goal.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_goal_agent \
    agent.goal_mode="${GOAL_MODE}" \
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
    experiment_name="${EXPERIMENT_NAME:-eval-pdms-teacher-200il_30goal-${GOAL_MODE}-general}" \
    worker=sequential

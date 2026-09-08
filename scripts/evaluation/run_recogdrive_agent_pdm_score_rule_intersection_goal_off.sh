set -x

# Goal-OFF PDMS counterpart of run_recogdrive_agent_pdm_score_rule_intersection_goal.sh.
# Same checkpoint / goal_mode / split, but no privileged GT goal is fed.
#
# Must use run_pdm_score_recogdrive.py (not the _goal entrypoint): the goal
# scorer always passes Scene, and compute_trajectory then extracts the GT
# endpoint even if strict_eval_goal=false. The non-goal scorer calls
# compute_trajectory(agent_input) with scene=None.
#
#   CHECKPOINT=/path/to.ckpt \
#     bash scripts/evaluation/run_recogdrive_agent_pdm_score_rule_intersection_goal_off.sh

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest_general_or_no_tag}"
#navtest_rule_intersection navtest_safety_dynamics_interaction navtest_progress_curbside_stopgo navtest_general_or_no_tag
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:$PATH" #nby
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/recdrive-scene"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-62665}
PORT=${PORT:-62664}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CHECKPOINT="${CHECKPOINT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp/training_teacher_general_or_no_tag_200il_30goal_adaln_newvlm/2026.09.07.09.20.11/lightning_logs/version_0/checkpoints/epoch=96-step=57230.ckpt}"
# GOAL_MODE must still match the mode the checkpoint was TRAINED with, so the
# goal weights load. They simply receive no goal tensor at inference.
GOAL_MODE="${GOAL_MODE:-adaln}"
METRIC_CACHE_PATH="/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache"

/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_goal_agent \
    agent.goal_mode="${GOAL_MODE}" \
    agent.strict_eval_goal=false \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path='/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name="${EXPERIMENT_NAME:-eval-pdms-teacher-mask-general-96epoch-goal-off}" \
    worker=sequential

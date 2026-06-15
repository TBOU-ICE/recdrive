#!/usr/bin/env bash
# 1-GPU debug run for Expert-Reward RL (strong expert IL anchor).
# Use this to verify the new agent loads, reward/loss values look sane,
# and no CUDA OOM before launching the full 8-GPU job.
set -euo pipefail

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/recdrive-opd-dit}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}"
CACHE_PATH="${CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train}"
EXPERT_DATA_PATH="${EXPERT_DATA_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/data/dataset_decoupled_v2_clean.pkl}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_expert_reward_rl_1gpu_debug}"
LOG_ROOT="${LOG_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/training_expert_reward_rl_debug}"

echo "[expert-reward-rl-debug] CHECKPOINT=${CHECKPOINT}"
echo "[expert-reward-rl-debug] EXPERT_DATA_PATH=${EXPERT_DATA_PATH}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=1 \
  --master_port=23470 \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_temporal_rl.py" \
  agent=recogdrive_agent_expert_reward_rl \
  agent.lr=1e-4 \
  agent.grpo=True \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  "agent.checkpoint_path='${CHECKPOINT}'" \
  "agent.reference_policy_checkpoint='${CHECKPOINT}'" \
  "agent.metric_cache_path='${METRIC_CACHE_PATH}'" \
  "agent.expert_data_path='${EXPERT_DATA_PATH}'" \
  agent.rl_pdms_weight=1.0 \
  agent.rl_temporal_reward_weight=0.1 \
  agent.expert_il_weight=0.5 \
  agent.rl_grpo_sample_time=4 \
  trainer.params.max_epochs=1 \
  trainer.params.precision=bf16-mixed \
  trainer.params.num_nodes=1 \
  trainer.params.devices=1 \
  dataloader.params.batch_size=2 \
  logger.type=tensorboard \
  "+logger.save_dir='${LOG_ROOT}/tensorboard'" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=False \
  force_cache_computation=False \
  +temporal_min_dt_s=0.1 \
  +temporal_max_dt_s=1.1 \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

#!/usr/bin/env bash
# Evaluate the OPD-trained 2B student at step 5000.
#
# Before running, merge the OPD checkpoint into a HF model directory:
#
#   python scripts/tools/merge_opd_checkpoint.py \
#     --base_model_path /mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/InternVL3-2B-ckpt400-merged \
#     --opd_ckpt_path   /workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/train_opd_sft_8gpu_oneimg/step00005000.pt \
#     --output_dir      /mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/InternVL3-2B-opd-step5000-merged
#
set -euo pipefail
set -x

TRAIN_TEST_SPLIT=navtest
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PATH="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/recdrive-opd-vlm"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"

MASTER_PORT=${MASTER_PORT:-63670}
export MASTER_PORT=${MASTER_PORT}

# ── Paths ─────────────────────────────────────────────────────────────────────
# VLM: merged HF directory (base + OPD step-5000 weights)
OPD_VLM_PATH="${OPD_VLM_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/InternVL3-2B-opd-step5000-merged}"

# DiT checkpoint: same Stage-3 RL checkpoint as baseline (action head unchanged by OPD)
CHECKPOINT="${CHECKPOINT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_dit_distill_2b/2026.04.23.10.40.50/lightning_logs/version_0/checkpoints/epoch=4-step=6650.ckpt}"

METRIC_CACHE_PATH="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache"

echo "OPD_VLM_PATH=${OPD_VLM_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    "agent.checkpoint_path=\"${CHECKPOINT}\"" \
    agent.vlm_path="${OPD_VLM_PATH}" \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    agent.metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name=recogdrive_agent_eval_opd_step5000 \
    worker=sequential

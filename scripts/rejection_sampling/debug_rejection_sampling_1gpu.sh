#!/bin/bash
# ---------------------------------------------------------------------------
# Single-GPU debug script for rejection sampling.
# Runs 20 scenes × 2 samples to verify the full pipeline end-to-end.
# No torchrun / MASTER_ADDR needed.
#
# Usage:
#   bash scripts/rejection_sampling/debug_rejection_sampling_1gpu.sh
#
# Optional overrides:
#   GPU=1 bash ...           # use GPU 1 instead of GPU 0
#   MAX_SCENES=5 bash ...    # run only 5 scenes
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------- env ----------
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/recdrive"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:/workspace/recdrive/internvl_chat${PYTHONPATH:+:${PYTHONPATH}}"

# ---------- GPU ----------
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ---------- paths ----------
VLM_PATH="${VLM_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-8B}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/rejection_sampling_debug}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
MAX_SCENES="${MAX_SCENES:-20}"

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo " [DEBUG] Rejection Sampling – 1 GPU"
echo " GPU=${GPU}"
echo " VLM_PATH=${VLM_PATH}"
echo " MAX_SCENES=${MAX_SCENES}  NUM_SAMPLES=2"
echo " OUTPUT_DIR=${OUTPUT_DIR}"
echo "=============================="

# Single-process launch (no torchrun): LOCAL_RANK/RANK/WORLD_SIZE default to 0/0/1
/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_rejection_sampling_vlm.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    hydra/job_logging=stdout \
    hydra.output_subdir=null \
    rs.vlm_path="${VLM_PATH}" \
    rs.metric_cache_path="${METRIC_CACHE_PATH}" \
    rs.output_dir="${OUTPUT_DIR}" \
    rs.max_scenes="${MAX_SCENES}" \
    rs.num_samples=2 \
    rs.pdm_threshold=0.4 \
    rs.plan_threshold=1.0 \
    rs.mot_pred_threshold=0.5 \
    rs.temperature=0.7 \
    rs.top_p=0.9 \
    rs.max_num_tiles=12 \
    rs.log_every=5

echo ""
echo "=============================="
echo " Output files:"
ls -lh "${OUTPUT_DIR}"/*.jsonl 2>/dev/null || echo "  (no .jsonl files found)"
echo ""

# Quick sanity check: count lines per file
echo " Line counts:"
for f in "${OUTPUT_DIR}"/*.jsonl; do
    [ -f "$f" ] && printf "  %-55s %d lines\n" "$(basename $f)" "$(wc -l < "$f")"
done

echo ""
echo " Sample record from traj_accepted_rank0.jsonl (first line):"
TRAJ_FILE="${OUTPUT_DIR}/traj_accepted_rank0.jsonl"
if [ -f "${TRAJ_FILE}" ] && [ -s "${TRAJ_FILE}" ]; then
    python -c "
import json, sys
with open('${TRAJ_FILE}') as f:
    rec = json.loads(f.readline())
print(json.dumps({k: rec[k] for k in ['id','question_type','status','best_reward','num_generated','num_valid','gt_answer']}, indent=2))
"
else
    echo "  (file empty or not found – all traj scenes may have failed/been rejected)"
fi

echo ""
echo " Sample record from plan_accepted_rank0.jsonl (first line):"
PLAN_FILE="${OUTPUT_DIR}/plan_accepted_rank0.jsonl"
if [ -f "${PLAN_FILE}" ] && [ -s "${PLAN_FILE}" ]; then
    python -c "
import json
with open('${PLAN_FILE}') as f:
    rec = json.loads(f.readline())
print(json.dumps({k: rec[k] for k in ['id','question_type','status','best_reward','num_generated','num_valid','gt_answer']}, indent=2))
"
else
    echo "  (file empty or not found)"
fi

echo "=============================="
echo " Debug run complete."

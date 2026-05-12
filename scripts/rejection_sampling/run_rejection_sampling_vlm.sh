#!/bin/bash
# ---------------------------------------------------------------------------
# Rejection sampling on the SFT-trained InternVL3 VLM.
#
# Three question types with deterministic ground-truth are sampled:
#   traj     – trajectory prediction  (PDM Score reward)
#   plan     – driving plan           (exact-match reward)
#   mot_pred – per-agent motion pred  (fraction-correct reward)
#
# Each GPU processes a shard of the training scenes independently.
# Output files per rank (9 total):
#   {traj|plan|mot_pred}_{accepted|rejected|failed}_rank{N}.jsonl
#
# Merge after all ranks finish:
#   for qt in traj plan mot_pred; do
#       for s in accepted rejected failed; do
#           cat ${OUT}/${qt}_${s}_rank*.jsonl > ${OUT}/${qt}_${s}_all.jsonl
#       done
#   done
# ---------------------------------------------------------------------------
set -euo pipefail

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/code"
export OPENSCENE_DATA_ROOT="/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# ---------- cluster / distributed settings ----------
NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is not set}"
MASTER_PORT="${MASTER_PORT:-23457}"
GPUS="${GPUS:-8}"

# ---------- paths ----------
VLM_PATH="${VLM_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-8B}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/rejection_sampling}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo " Rejection Sampling VLM"
echo " NNODES=${NNODES}  RANK=${RANK}  GPUS=${GPUS}"
echo " VLM_PATH=${VLM_PATH}"
echo " OUTPUT_DIR=${OUTPUT_DIR}"
echo "=============================="

/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_rejection_sampling_vlm.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    hydra/job_logging=stdout \
    hydra.output_subdir=null \
    rs.vlm_path="${VLM_PATH}" \
    rs.metric_cache_path="${METRIC_CACHE_PATH}" \
    rs.output_dir="${OUTPUT_DIR}" \
    rs.num_samples=8 \
    rs.pdm_threshold=0.4 \
    rs.plan_threshold=1.0 \
    rs.mot_pred_threshold=0.5 \
    rs.temperature=0.7 \
    rs.top_p=0.9 \
    rs.max_num_tiles=12

echo "Shard files written to ${OUTPUT_DIR}"
echo "Merging shards …"
for qt in traj plan mot_pred; do
    for s in accepted rejected failed; do
        cat "${OUTPUT_DIR}/${qt}_${s}_rank"*.jsonl > "${OUTPUT_DIR}/${qt}_${s}_all.jsonl"
        echo "Merged → ${OUTPUT_DIR}/${qt}_${s}_all.jsonl"
    done
done
echo "Done."

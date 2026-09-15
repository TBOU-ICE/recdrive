#!/usr/bin/env bash
# Phase A -- cache privileged-teacher trajectories for teacher-rollout SFT.
#
# Routes every sample to the bucket teacher that owns it (same routing OPD uses),
# rolls that teacher out under the clean GT goal with DDIM eta=0, and writes one
# teacher-rollout-v1 payload.
#
# Data comes straight from the original /mnt caches via the prebuilt
# direct-absolute-paths indexes: no symlink farm, and no hard link (which could
# not cross the PolarFS -> NAS boundary anyway).
#
# Every artefact lands under OUTPUT_ROOT (/workspace/output/tensorboard).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export PATH="${CONDA_BIN:-/opt/conda/envs/recdrive/bin}:$PATH"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/output/tensorboard}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${OUTPUT_ROOT}}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-teacher_rollout_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/rollout.log}"
mkdir -p "${OUTPUT_DIR}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23611}"

# Prebuilt exclusive bucket indexes; their union is the full navtrain + SimScale
# trainable mixture and already encodes the teacher routing.
INDEX_ROOT="${INDEX_ROOT:-/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm_direct}"
INDEX_NAME="${INDEX_NAME:-train_index.json}"

TEACHER_ROOT="${TEACHER_ROOT:-/mnt/datasets/recdrive/20260513/exp2/exp}"
TEACHER_GENERAL="${TEACHER_GENERAL:-${TEACHER_ROOT}/training_teacher_general_or_no_tag_200il_30goal_adaln_newvlm/2026.09.08.07.58.19/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"
TEACHER_PROGRESS="${TEACHER_PROGRESS:-${TEACHER_ROOT}/training_teacher_progress_curbside_stopgo_200il_30goal_adaln_newvlm/2026.09.08.07.52.51/lightning_logs/version_0/checkpoints/epoch=197-step=91872.ckpt}"
TEACHER_RULE="${TEACHER_RULE:-${TEACHER_ROOT}/training_teacher_rule_intersection_200il_30goal_adaln_newvlm/2026.09.08.07.52.51/lightning_logs/version_0/checkpoints/epoch=198-step=62486.ckpt}"
TEACHER_SAFETY="${TEACHER_SAFETY:-${TEACHER_ROOT}/training_teacher_safety_dynamics_interaction_200il_30goal_adaln_newvlm/2026.09.08.07.58.36/lightning_logs/version_0/checkpoints/epoch=198-step=38606.ckpt}"

GOAL_MODE="${GOAL_MODE:-adaln}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# 0 keeps only the deterministic rollout. Set to 4 to also cache K stochastic
# samples per token for the multi-modal SFT ablation (~5x generation time).
NUM_RANDOM_SAMPLES="${NUM_RANDOM_SAMPLES:-0}"
LIMIT="${LIMIT:-0}"

for f in "${TEACHER_GENERAL}" "${TEACHER_PROGRESS}" "${TEACHER_RULE}" "${TEACHER_SAFETY}"; do
  [[ -f "${f}" ]] || { echo "[ERROR] teacher checkpoint missing: ${f}" >&2; exit 1; }
done
for b in general_or_no_tag progress_curbside_stopgo rule_intersection safety_dynamics_interaction; do
  [[ -f "${INDEX_ROOT}/${b}/${INDEX_NAME}" ]] || {
    echo "[ERROR] direct index missing: ${INDEX_ROOT}/${b}/${INDEX_NAME}" >&2; exit 1; }
done

echo "======================================================================"
echo "[teacher-rollout] INDEX_ROOT=${INDEX_ROOT}  (direct absolute paths, no links)"
echo "[teacher-rollout] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[teacher-rollout] GOAL_MODE=${GOAL_MODE}  (clean GT goal, DDIM eta=0, teachers in eval())"
echo "[teacher-rollout] NUM_RANDOM_SAMPLES=${NUM_RANDOM_SAMPLES}  BATCH_SIZE=${BATCH_SIZE}  GPUS=${GPUS}"
echo "======================================================================"

TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/recdrive/bin/torchrun}"
EXTRA_ARGS=()
[[ "${NUM_RANDOM_SAMPLES}" != "0" ]] && EXTRA_ARGS+=( --num-random-samples "${NUM_RANDOM_SAMPLES}" )
[[ "${LIMIT}" != "0" ]] && EXTRA_ARGS+=( --limit "${LIMIT}" )

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/scripts/generation/run_teacher_rollout_cache.py" \
  --index-root "${INDEX_ROOT}" \
  --index-name "${INDEX_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --teacher-general "${TEACHER_GENERAL}" \
  --teacher-progress "${TEACHER_PROGRESS}" \
  --teacher-rule "${TEACHER_RULE}" \
  --teacher-safety "${TEACHER_SAFETY}" \
  --goal-mode "${GOAL_MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "${LOG_FILE}"

echo "[teacher-rollout] done. Check ${OUTPUT_DIR}/teacher_rollout_summary.json"
echo "[teacher-rollout] per-bucket fde_mean_m there should match the OPD log's fde_gt_teacher_final_m."

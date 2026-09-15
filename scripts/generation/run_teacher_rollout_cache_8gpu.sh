#!/usr/bin/env bash
# Phase A: generate the offline teacher-rollout cache that Phase B SFTs on.
#
# Each token is routed to its scenario expert, the teacher is conditioned on the
# privileged GT goal, and K+1 DDIM rollouts are written out. Nothing is written
# into the (read-only / Alluxio-backed) feature cache: shards land under
# TEACHER_ROLLOUT_DIR, and the caches are read in place through their direct
# indexes -- no symlinks, no hardlinks, no copying.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/opt/conda/envs/recdrive/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/output/tensorboard}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SCENE_ROUTER_BAD_CACHE_LIST="${SCENE_ROUTER_BAD_CACHE_LIST:-${REPO_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"

NNODES="${NNODES:-1}"; RANK="${RANK:-0}"; GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-23531}"

TEACHER_ROLLOUT_DIR="${TEACHER_ROLLOUT_DIR:-${NAVSIM_EXP_ROOT}/teacher_rollout_cache_adaln}"

# The four 200-epoch adaLN goal teachers.
TEACHER_ROOT="${TEACHER_ROOT:-/mnt/datasets/recdrive/20260513/exp2/exp}"
TEACHER_GENERAL_CKPT="${TEACHER_GENERAL_CKPT:-${TEACHER_ROOT}/training_teacher_general_or_no_tag_200il_30goal_adaln_newvlm/2026.09.08.07.58.19/lightning_logs/version_0/checkpoints/epoch=199-step=118000.ckpt}"
TEACHER_PROGRESS_CKPT="${TEACHER_PROGRESS_CKPT:-${TEACHER_ROOT}/training_teacher_progress_curbside_stopgo_200il_30goal_adaln_newvlm/2026.09.08.07.52.51/lightning_logs/version_0/checkpoints/epoch=197-step=91872.ckpt}"
TEACHER_RULE_CKPT="${TEACHER_RULE_CKPT:-${TEACHER_ROOT}/training_teacher_rule_intersection_200il_30goal_adaln_newvlm/2026.09.08.07.52.51/lightning_logs/version_0/checkpoints/epoch=198-step=62486.ckpt}"
TEACHER_SAFETY_CKPT="${TEACHER_SAFETY_CKPT:-${TEACHER_ROOT}/training_teacher_safety_dynamics_interaction_200il_30goal_adaln_newvlm/2026.09.08.07.58.36/lightning_logs/version_0/checkpoints/epoch=198-step=38606.ckpt}"
TEACHER_GOAL_MODE="${TEACHER_GOAL_MODE:-adaln}"

# Direct indexes built by scripts/data/prep_bucket_il_direct_indexes.py -- the
# same ones the IL bucket experts train on. Each <bucket>/train_index.json lists
# absolute token-dir paths with navtrain + simscale already merged, so the bucket
# routing comes for free and no manifest / quality whitelist is involved.
DIRECT_INDEX_ROOT="${DIRECT_INDEX_ROOT:-/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm_direct}"
INDEX_NAME="${INDEX_NAME:-train_index.json}"

# 0 = one deterministic rollout per scene. That matches both how the teacher was
# trained (a single target per scene at every noise level) and what the OPD KD
# target is (deterministic=True). Raise it only to run the goal-space-coverage
# ablation, after checking that sample spread is actually non-trivial.
NUM_SAMPLES="${NUM_SAMPLES:-0}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP="${AMP:-bf16}"                  # matches the OPD trainer's bf16-mixed precision
LIMIT="${LIMIT:-0}"                 # >0 for a smoke test

mkdir -p "${TEACHER_ROLLOUT_DIR}"
LOG_FILE="${LOG_FILE:-${TEACHER_ROLLOUT_DIR}/rollout.log}"

echo "[TeacherRollout] out=${TEACHER_ROLLOUT_DIR} K=${NUM_SAMPLES} amp=${AMP}"
echo "[TeacherRollout] direct_index_root=${DIRECT_INDEX_ROOT}"

torchrun --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/scripts/generation/run_teacher_rollout_cache.py" \
  --out-dir "${TEACHER_ROLLOUT_DIR}" \
  --direct-index-root "${DIRECT_INDEX_ROOT}" --index-name "${INDEX_NAME}" \
  --teacher-general-or-no-tag "${TEACHER_GENERAL_CKPT}" \
  --teacher-progress-curbside-stopgo "${TEACHER_PROGRESS_CKPT}" \
  --teacher-rule-intersection "${TEACHER_RULE_CKPT}" \
  --teacher-safety-dynamics-interaction "${TEACHER_SAFETY_CKPT}" \
  --teacher-goal-mode "${TEACHER_GOAL_MODE}" \
  --num-samples "${NUM_SAMPLES}" --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --amp "${AMP}" --limit "${LIMIT}" 2>&1 | tee "${LOG_FILE}"

echo "[TeacherRollout] done. Shards:"
ls -la "${TEACHER_ROLLOUT_DIR}"/teacher_rollout_shard_*.pt
echo "[TeacherRollout] Before Phase B: check the per-bucket canonical FDE above against"
echo "                 fde_gt_teacher_final_m in the OPD log. A mismatch means the routing"
echo "                 or the goal binding is wrong and the labels are not trustworthy."

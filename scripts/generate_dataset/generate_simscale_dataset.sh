#!/usr/bin/env bash
# Generate InternVL SFT data (trajectory QA + rule-based QA) from SimScale scenes.
#
# Mirrors the ReCogDrive data pipeline but targets the SimScale split and needs
# NO external VLM for the default (phase-1) rule-based generation.
#
# Round selection:
#   ROUND=0 (default) -> synthetic_reaction_pdm_v1.0-0 / train_test_split=simscale_pdm_round0
#   ROUND=1           -> synthetic_reaction_pdm_v1.0-1 / train_test_split=simscale_pdm_round1
#
# Parallelism: launches SHARDS local CPU processes, each handling a disjoint
# subset of logs. Outputs one pair of jsonl files per shard, then prints the
# merge commands.
#
# Usage:
#   bash scripts/generate_dataset/generate_simscale_dataset.sh
#   ROUND=1 SHARDS=32 bash scripts/generate_dataset/generate_simscale_dataset.sh
#   SIMSCALE_MAX_SCENES=20 SHARDS=1 bash scripts/generate_dataset/generate_simscale_dataset.sh   # smoke test
#   EMIT=traj bash scripts/generate_dataset/generate_simscale_dataset.sh                          # traj QA only
set -euo pipefail

ROUND="${ROUND:-0}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-simscale_pdm_round${ROUND}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIMSCALE_ROOT="${SIMSCALE_ROOT:-/mnt/datasets/simscale/20260709}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${SIMSCALE_ROOT}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SIMSCALE_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# CPU-only workload. Pin each process to a single math thread so that launching
# many shards does NOT oversubscribe the cores (each shard is one python proc).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"   # generation is CPU-only; keep GPUs idle

# Image paths in the jsonl are stored relative to this root (== meta `root`).
export SIMSCALE_IMAGE_ROOT="${SIMSCALE_IMAGE_ROOT:-${SIMSCALE_ROOT}}"
export SIMSCALE_QA_OUT_DIR="${SIMSCALE_QA_OUT_DIR:-${SIMSCALE_ROOT}/simscale_vlm_qa/${DATASET_NAME}}"
export SIMSCALE_ROOT
export SIMSCALE_EMIT="${EMIT:-both}"
export SIMSCALE_MAX_SCENES="${SIMSCALE_MAX_SCENES:-0}"
export SIMSCALE_MAX_LOGS="${SIMSCALE_MAX_LOGS:-0}"
export VRU_KEEP_EMPTY_PROB="${VRU_KEEP_EMPTY_PROB:-0.1}"
export USE_VLM="${USE_VLM:-0}"
export RESUME="${RESUME:-0}"
# Quality control: keep only high-PDMS scenes (== downstream `_quality` subset).
#   QUALITY_FILTER=1 (default) uses the pre-baked allowlist for the round.
#   PDMS_THRESHOLD>0 selects tokens with PDMS>=threshold from the scores CSV.
#   SIMSCALE_ALLOWLIST=<path> overrides with a custom allowlist file.
export QUALITY_FILTER="${QUALITY_FILTER:-1}"
export PDMS_THRESHOLD="${PDMS_THRESHOLD:-0}"
export SIMSCALE_ALLOWLIST="${SIMSCALE_ALLOWLIST:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/recdrive/bin/python}"
# CPU-bound: default parallelism to ~1/3 of cores (each proc also does IO/pkl load),
# capped at 64 to bound RAM (each proc imports torch ~2GB) and FS metadata pressure.
_NPROC="$(nproc 2>/dev/null || echo 16)"
_DEFAULT_SHARDS=$(( _NPROC / 3 )); [[ "${_DEFAULT_SHARDS}" -lt 1 ]] && _DEFAULT_SHARDS=1
[[ "${_DEFAULT_SHARDS}" -gt 64 ]] && _DEFAULT_SHARDS=64
SHARDS="${SHARDS:-${_DEFAULT_SHARDS}}"
SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_generate_dataset_simscale.py"

mkdir -p "${SIMSCALE_QA_OUT_DIR}" "${SIMSCALE_ROOT}/logs"

echo "=================================================="
echo " SimScale VLM data generation"
echo "   ROUND=${ROUND}  DATASET_NAME=${DATASET_NAME}"
echo "   TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT}"
echo "   OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "   OUT_DIR=${SIMSCALE_QA_OUT_DIR}"
echo "   EMIT=${SIMSCALE_EMIT}  SHARDS=${SHARDS}  USE_VLM=${USE_VLM}"
echo "   MAX_SCENES(per shard)=${SIMSCALE_MAX_SCENES}  RESUME=${RESUME}"
echo "   QUALITY_FILTER=${QUALITY_FILTER}  PDMS_THRESHOLD=${PDMS_THRESHOLD}  ALLOWLIST=${SIMSCALE_ALLOWLIST:-<pre-baked>}"
echo "=================================================="

pids=()
for ((i=0; i<SHARDS; i++)); do
  SHARD_INDEX="${i}" SHARD_COUNT="${SHARDS}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
      train_test_split="${TRAIN_TEST_SPLIT}" \
      experiment_name="generate_dataset_simscale_${DATASET_NAME}" \
      hydra/job_logging=stdout \
      hydra.output_subdir=null \
      > "${SIMSCALE_ROOT}/logs/gen_simscale_${DATASET_NAME}_shard${i}of${SHARDS}.log" 2>&1 &
  pids+=("$!")
  echo "launched shard ${i}/${SHARDS} pid=${pids[-1]} log=${SIMSCALE_ROOT}/logs/gen_simscale_${DATASET_NAME}_shard${i}of${SHARDS}.log"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
    echo "[ERROR] shard pid=${pid} failed"
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "[ERROR] one or more shards failed; inspect logs in ${SIMSCALE_ROOT}/logs/"
  exit 1
fi

echo "--------------------------------------------------"
echo "All shards done."

# Auto-merge shards into *_all.jsonl (the paths the meta points to) using an
# allowlist-aware, dedup merger. Robust to stale/mixed shards from earlier runs
# (RESUME can leave pre-QC records behind). Idempotent. Disable with MERGE=0.
MERGE="${MERGE:-1}"
if [[ "${MERGE}" == "1" ]]; then
  KINDS="${SIMSCALE_EMIT}"
  [[ "${SIMSCALE_EMIT}" == "both" ]] && KINDS="traj,qa"
  ROUND="${ROUND}" \
  SIMSCALE_ROOT="${SIMSCALE_ROOT}" \
  SIMSCALE_QA_OUT_DIR="${SIMSCALE_QA_OUT_DIR}" \
  QUALITY_FILTER="${QUALITY_FILTER}" \
  PDMS_THRESHOLD="${PDMS_THRESHOLD}" \
  SIMSCALE_ALLOWLIST="${SIMSCALE_ALLOWLIST}" \
  KINDS="${KINDS}" \
  "${PYTHON_BIN}" "${NAVSIM_DEVKIT_ROOT}/scripts/generate_dataset/merge_simscale_qa.py"
  echo "[merge] *_all.jsonl paths match the meta in shell/data_info/recogdrive_simscale_*.json"
else
  echo "MERGE=0 -> skipped. Merge later with:"
  echo "  ROUND=${ROUND} SIMSCALE_ROOT=${SIMSCALE_ROOT} ${PYTHON_BIN} ${NAVSIM_DEVKIT_ROOT}/scripts/generate_dataset/merge_simscale_qa.py"
fi

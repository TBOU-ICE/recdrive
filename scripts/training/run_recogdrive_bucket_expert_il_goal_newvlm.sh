#!/usr/bin/env bash
set -euo pipefail

# Shared goal-conditioned bucket IL teacher. Wrappers set BUCKET_NAME / BUCKET_FILE
# and optionally BASE_CKPT, then exec this file.
#
# Mixture: same navtrain+SimScale bucket views as the no-goal IL teachers.
# Agent: recogdrive_goal_agent (privileged GT 8th waypoint, default adaln).
# Init: weight-only BASE_CKPT (no-goal bucket IL). RESUME_CKPT = Lightning resume.
#
# GPU idle kill: do NOT walk Alluxio on the GPU job. Reuse the prebuilt
# il_training_newvlm indexes (same tokens/caches as no-goal IL):
#   SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_general_il_goal_newvlm.sh

if [[ -z "${BUCKET_NAME:-}" || -z "${BUCKET_FILE:-}" ]]; then
  echo "[ERROR] BUCKET_NAME and BUCKET_FILE must be set." >&2
  echo "  Use one of the run_recogdrive_bucket_expert_*_il_goal_newvlm.sh wrappers." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"

export PATH="${CONDA_BIN:-/opt/conda/envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-bd-su01/nby/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CACHE_READ_MAX_RETRIES="${CACHE_READ_MAX_RETRIES:-10}"
export CACHE_READ_RETRY_BASE_SEC="${CACHE_READ_RETRY_BASE_SEC:-0.5}"
export BAD_CACHE_LIST="${BAD_CACHE_LIST:-${NAVSIM_DEVKIT_ROOT}/data/epdms/bad_cache_shards_newvlm.txt}"
export RESILIENT_CACHE_LOADING="${RESILIENT_CACHE_LOADING:-1}"
export CACHE_LOAD_MAX_RETRIES="${CACHE_LOAD_MAX_RETRIES:-8}"
export CACHE_LOAD_TIMEOUT_SEC="${CACHE_LOAD_TIMEOUT_SEC:-60}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23574}"
GPUS="${GPUS:-8}"

GOAL_MODE="${GOAL_MODE:-adaln}"
# Robust-teacher goal corruption, disjoint by construction (see encode_goal):
# dropout=0.10 + noise=0.20 -> 10% masked / 20% noisy / 70% clean GT goal.
# Set both to 0 to reproduce the old pure-privileged (100% clean goal) behaviour.
GOAL_DROPOUT_P="${GOAL_DROPOUT_P:-0.1}"
GOAL_NOISE_P="${GOAL_NOISE_P:-0.2}"
GOAL_NOISE_STD_XY="${GOAL_NOISE_STD_XY:-1.0}"
GOAL_NOISE_STD_HEADING="${GOAL_NOISE_STD_HEADING:-0.1}"
VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
BASE_CKPT="${BASE_CKPT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[teacher-il-goal] RESUME full state from: ${RESUME_CKPT}"
  CKPT_ARGS=( "agent.checkpoint_path=null" "ckpt_path='${RESUME_CKPT}'" )
elif [[ -n "${BASE_CKPT}" ]]; then
  echo "[teacher-il-goal] FRESH warm-start from BASE_CKPT: ${BASE_CKPT}"
  CKPT_ARGS=( "agent.checkpoint_path='${BASE_CKPT}'" )
else
  echo "[teacher-il-goal] FRESH training from a RANDOM-INIT DiT"
  CKPT_ARGS=( "agent.checkpoint_path=null" )
fi

LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/mnt/datasets/simscale/20260709/data/simscale/new_vlm_quality_views}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/mnt/datasets/simscale/20260709/data/simscale}"
NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR:-/mnt/datasets/simscale/20260709/data/navtrain_scene/output/navtrain}"
SIM_ROUNDS="${SIM_ROUNDS:-${SIM_ROUND:-0,1}}"

# Same views as no-goal IL. Goal vs no-goal is the agent, not the cache tree.
MIX_ROOT="${MIX_ROOT:-/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm/${BUCKET_NAME}_fullmix}"
NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH:-${MIX_ROOT}/navtrain_bucket_cache}"
SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH:-${MIX_ROOT}/simscale_bucket_cache}"
MIX_INFO_DIR="${MIX_INFO_DIR:-${MIX_ROOT}/metadata}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_teacher_${BUCKET_NAME}_200il_30goal_${GOAL_MODE}_newvlm}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/recdrive/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/recdrive/bin/python}"
PREP_ONLY="${PREP_ONLY:-false}"
SKIP_PREP="${SKIP_PREP:-false}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-true}"
KEEPALIVE_PID=""

stop_gpu_keepalive() {
  if [[ -n "${KEEPALIVE_PID}" ]] && kill -0 "${KEEPALIVE_PID}" 2>/dev/null; then
    echo "[teacher-il-goal] stopping GPU keepalive pid=${KEEPALIVE_PID}"
    kill "${KEEPALIVE_PID}" 2>/dev/null || true
    wait "${KEEPALIVE_PID}" 2>/dev/null || true
  fi
  KEEPALIVE_PID=""
}
trap stop_gpu_keepalive EXIT

start_gpu_keepalive() {
  if [[ "${GPU_KEEPALIVE}" != "true" ]]; then
    echo "[teacher-il-goal] GPU keepalive disabled (GPU_KEEPALIVE=${GPU_KEEPALIVE})"
    return 0
  fi
  if [[ "${PREP_ONLY}" == "true" ]]; then
    echo "[teacher-il-goal] PREP_ONLY=true; skip GPU keepalive"
    return 0
  fi
  echo "[teacher-il-goal] starting GPU keepalive on ${GPUS} device(s) to avoid idle eviction"
  GPU_KEEPALIVE_N="${GPUS}" "${PYTHON_BIN}" - <<'PYKEEP' &
import os
import time
import torch

n = max(1, int(os.environ.get("GPU_KEEPALIVE_N", "1")))
n = min(n, torch.cuda.device_count()) if torch.cuda.is_available() else 0
if n == 0:
    raise SystemExit(0)
bufs = []
for i in range(n):
    t = torch.ones((2048, 2048), device=f"cuda:{i}", dtype=torch.float32)
    bufs.append(t)
print(f"[gpu-keepalive] holding {n} GPU(s)", flush=True)
while True:
    for t in bufs:
        t.mul_(1.0000001)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time.sleep(3)
PYKEEP
  KEEPALIVE_PID="$!"
}

start_gpu_keepalive

if [[ ! -d "${NAV_CACHE_PATH}" ]]; then
  echo "[ERROR] NAV_CACHE_PATH does not exist: ${NAV_CACHE_PATH}" >&2
  exit 1
fi
if [[ ! -d "${VLM_PATH}" ]]; then
  echo "[ERROR] VLM_PATH does not exist: ${VLM_PATH}" >&2
  exit 1
fi
if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[ERROR] RESUME_CKPT does not exist: ${RESUME_CKPT}" >&2
  exit 1
fi
if [[ -z "${RESUME_CKPT}" && -n "${BASE_CKPT}" && ! -f "${BASE_CKPT}" ]]; then
  echo "[ERROR] BASE_CKPT does not exist: ${BASE_CKPT}" >&2
  exit 1
fi

PREP_PY="${NAVSIM_DEVKIT_ROOT}/scripts/data/prep_bucket_il_newvlm.py"
TRAIN_INDEX_PATH="${TRAIN_INDEX_PATH:-${MIX_INFO_DIR}/train_index.json}"
VAL_INDEX_PATH="${VAL_INDEX_PATH:-/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm/navtrain_cache_index.json}"
SUMMARY_JSON="${MIX_INFO_DIR}/bucket_il_sources_summary.json"

if [[ "${SKIP_PREP}" == "true" ]]; then
  if [[ ! -f "${SUMMARY_JSON}" ]]; then
    echo "[ERROR] SKIP_PREP=true but summary missing: ${SUMMARY_JSON}" >&2
    exit 1
  fi
  echo "[teacher-il-goal] SKIP_PREP=true; reuse existing bucket views at ${MIX_ROOT}"
elif [[ -f "${SUMMARY_JSON}" && -f "${TRAIN_INDEX_PATH}" ]]; then
  echo "[teacher-il-goal] bucket views + train index already present; skip linking"
else
  echo "[teacher-il-goal] building bucket symlinks + train index via ${PREP_PY}"
  "${PYTHON_BIN}" "${PREP_PY}" \
    --bucket-file "${BUCKET_FILE}" \
    --navtrain-output-dir "${NAVTRAIN_OUTPUT_DIR}" \
    --nav-cache "${NAV_CACHE_PATH}" \
    --sim-agent-cache-root "${SIM_AGENT_CACHE_ROOT}" \
    --sim-quality-cache-root "${SIM_QUALITY_CACHE_ROOT}" \
    --simscale-bucket-root "${SIMSCALE_BUCKET_ROOT}" \
    --mix-root-parent "$(dirname "${MIX_ROOT}")" \
    --sim-rounds "${SIM_ROUNDS}" \
    --nav-index-path "${VAL_INDEX_PATH}" \
    --workers "${PREP_WORKERS:-4}"
fi

if [[ "${PREP_ONLY}" == "true" ]]; then
  echo "[teacher-il-goal] PREP_ONLY=true; bucket views ready at ${MIX_ROOT}. Exiting before torchrun."
  exit 0
fi

echo "======================================================================"
echo "[teacher-il-goal] BUCKET_NAME=${BUCKET_NAME}  BUCKET_FILE=${BUCKET_FILE}"
echo "[teacher-il-goal] GOAL_MODE=${GOAL_MODE}  (privileged goal point = GT 8th waypoint)"
echo "[teacher-il-goal] goal corruption: clean=$("${PYTHON_BIN}" -c "print(1 - ${GOAL_DROPOUT_P} - ${GOAL_NOISE_P})") masked(GOAL_DROPOUT_P)=${GOAL_DROPOUT_P} noisy(GOAL_NOISE_P)=${GOAL_NOISE_P} noise_std_xy=${GOAL_NOISE_STD_XY}m noise_std_heading=${GOAL_NOISE_STD_HEADING}rad"
echo "[teacher-il-goal] MIXTURE: reuse no-goal IL views (navtrain_bucket + simscale_bucket)"
echo "[teacher-il-goal] BASE_CKPT=${BASE_CKPT:-<random-init DiT>}"
echo "[teacher-il-goal] RESUME_CKPT=${RESUME_CKPT:-<none>}"
echo "[teacher-il-goal] VLM_PATH=${VLM_PATH}"
echo "[teacher-il-goal] MIX_ROOT=${MIX_ROOT}"
echo "[teacher-il-goal] TRAIN_INDEX_PATH=${TRAIN_INDEX_PATH}"
echo "[teacher-il-goal] VAL_INDEX_PATH=${VAL_INDEX_PATH}"
echo "[teacher-il-goal] LR=${LR}  MAX_EPOCHS=${MAX_EPOCHS}  BATCH_SIZE=${BATCH_SIZE}"
echo "[teacher-il-goal] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[teacher-il-goal] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "======================================================================"

if [[ ! -f "${TRAIN_INDEX_PATH}" ]]; then
  echo "[ERROR] train index missing: ${TRAIN_INDEX_PATH}" >&2
  echo "  Build it first: bash scripts/training/prep_all_bucket_il_newvlm.sh" >&2
  echo "  Or reuse: MIX_ROOT=/mnt/datasets/simscale/20260709/data/simscale/il_training_newvlm/${BUCKET_NAME}_fullmix SKIP_PREP=true" >&2
  exit 1
fi

INDEX_ARGS=( "mixed_cache.index_path='${TRAIN_INDEX_PATH}'" )
if [[ -f "${VAL_INDEX_PATH}" ]]; then
  INDEX_ARGS+=( "cache_index_path='${VAL_INDEX_PATH}'" )
else
  echo "[teacher-il-goal] WARN: val index missing (${VAL_INDEX_PATH}); validation will walk NAV_CACHE_PATH"
fi

stop_gpu_keepalive

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  agent=recogdrive_goal_agent \
  agent.goal_mode="${GOAL_MODE}" \
  agent.goal_dropout_p="${GOAL_DROPOUT_P}" \
  agent.goal_noise_p="${GOAL_NOISE_P}" \
  agent.goal_noise_std_xy="${GOAL_NOISE_STD_XY}" \
  agent.goal_noise_std_heading="${GOAL_NOISE_STD_HEADING}" \
  "${CKPT_ARGS[@]}" \
  agent.lr="${LR}" \
  agent.grpo=False \
  agent.vlm_path="${VLM_PATH}" \
  agent.cam_type='single' \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.precision=bf16-mixed \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  trainer.params.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.params.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  cache_path="${NAV_CACHE_PATH}" \
  use_cache_without_dataset=true \
  force_cache_computation=False \
  use_mixed_cache=true \
  mixed_cache.paths="[${NAV_BUCKET_CACHE_PATH},${SIM_BUCKET_CACHE_PATH}]" \
  mixed_cache.names="[navtrain_bucket,simscale_bucket]" \
  mixed_cache.fullmix=true \
  "${INDEX_ARGS[@]}" \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

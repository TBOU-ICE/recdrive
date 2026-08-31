#!/usr/bin/env bash
# Full-mix navtrain + SimScale DiT IL launcher.
#
# Unlike run_recogdrive_train_mixed_simscale_2b.sh (weighted ratio sampling),
# this script sets mixed_cache.fullmix=true so every cached sample has equal
# probability. The effective batch composition follows dataset size
# (navtrain ~N + simscale round0 ~M0 + simscale round1 ~M1), not a fixed ratio.
#
# Representation contract (same as the scene-router / goal-teacher scripts):
#   VLM_PATH  = vlm_simscale_lora_merged
#   nav cache = new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_train
#   sim cache = new_vlm_hidden_state_nav_sim/recogdrive_agent_cache_dir_*
# Do not mix these with vlm_simscale_lora_vit_merged or the old
# /workspace/models/recdrive/v1.0.0/recogdrive_agent_cache_dir_train tree.
# The quality views at SIMSCALE_ROOT root symlink to the old-VLM cache; this
# script never reads those.
#
# By default both SimScale round0 and round1 quality-filtered caches are used.
# Override SIM_ROUNDS (comma-separated, e.g. "0" or "0,1") as needed.
#
# Usage:
#   bash scripts/training/run_recogdrive_train_fullmix_simscale_2b.sh
#   GPUS=8 MAX_EPOCHS=20 bash scripts/training/run_recogdrive_train_fullmix_simscale_2b.sh
#   SIM_ROUNDS=0,1 bash scripts/training/run_recogdrive_train_fullmix_simscale_2b.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PATH="${CONDA_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
# Soften Alluxio/FUSE flaky reads in dataset.py load_feature_target_from_pickle.
export CACHE_READ_MAX_RETRIES="${CACHE_READ_MAX_RETRIES:-10}"
export CACHE_READ_RETRY_BASE_SEC="${CACHE_READ_RETRY_BASE_SEC:-0.5}"

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/datasets/simscale/20260709}"
# New-VLM (LLM-only LoRA) hidden-state caches — same tree as scene-router.
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-${SIMSCALE_ROOT}/new_vlm_hidden_state_nav_sim}"
# Allowlists live here. Root-level *_quality dirs are old-VLM; do not train on them.
ALLOWLIST_ROOT="${ALLOWLIST_ROOT:-${SIMSCALE_ROOT}}"
# CPFS fallback for quality symlink views when a round has no *_quality next to source.
QUALITY_CACHE_ROOT="${QUALITY_CACHE_ROOT:-${SIMSCALE_ROOT}/data/simscale/new_vlm_quality_views}"
SIM_ROUNDS="${SIM_ROUNDS:-0,1}"
USE_QUALITY_CACHE="${USE_QUALITY_CACHE:-true}"

NAV_CACHE_PATH="${NAV_CACHE_PATH:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"

IFS=',' read -r -a SIM_ROUND_LIST <<< "${SIM_ROUNDS}"
SIM_CACHE_PATHS=()
SIM_CACHE_NAMES=()
PREP_ROUNDS=""
FORCE_PREP_QUALITY_CACHE="${FORCE_PREP_QUALITY_CACHE:-false}"
for round in "${SIM_ROUND_LIST[@]}"; do
  round="${round//[[:space:]]/}"
  if [[ -z "${round}" ]]; then
    continue
  fi
  dataset_name="synthetic_reaction_pdm_v1.0-${round}"
  if [[ "${USE_QUALITY_CACHE}" == "true" ]]; then
    src_quality="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}_quality"
    dst_quality="${QUALITY_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}_quality"
    if [[ "${FORCE_PREP_QUALITY_CACHE}" != "true" ]] \
       && [[ -d "${src_quality}" ]] && [[ -n "$(ls -A "${src_quality}" 2>/dev/null || true)" ]]; then
      sim_cache_path="${src_quality}"
    else
      sim_cache_path="${dst_quality}"
      if [[ "${FORCE_PREP_QUALITY_CACHE}" == "true" ]] \
         || [[ ! -d "${dst_quality}" ]] || [[ -z "$(ls -A "${dst_quality}" 2>/dev/null || true)" ]]; then
        PREP_ROUNDS="${PREP_ROUNDS}${PREP_ROUNDS:+,}${round}"
      fi
    fi
  else
    sim_cache_path="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}"
  fi
  SIM_CACHE_PATHS+=("${sim_cache_path}")
  SIM_CACHE_NAMES+=("simscale_r${round}")
done

if [[ ${#SIM_CACHE_PATHS[@]} -eq 0 ]]; then
  echo "[ERROR] SIM_ROUNDS resolved to no cache paths: ${SIM_ROUNDS}" >&2
  exit 1
fi
# Weight-only warm start (agent.initialize). Leave empty when using CKPT_PATH resume.
BASE_CKPT="${BASE_CKPT:-}"
VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23457}"
GPUS="${GPUS:-8}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_student_dit_il_all_data_newvlm}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/python}"
# Lightning full-state resume (model + optimizer + epoch/step). Default: new-VLM IL epoch=2.
CKPT_PATH="${CKPT_PATH:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"

if [[ "${USE_QUALITY_CACHE}" == "true" ]] && [[ -n "${PREP_ROUNDS}" ]]; then
  export PREP_SRC_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT}"
  export PREP_QUALITY_CACHE_ROOT="${QUALITY_CACHE_ROOT}"
  export PREP_ALLOWLIST_ROOT="${ALLOWLIST_ROOT}"
  export PREP_SIM_ROUNDS="${PREP_ROUNDS}"
  "${PYTHON_BIN}" - <<'PYPREP'
import os
from pathlib import Path

src_root = Path(os.environ["PREP_SRC_CACHE_ROOT"])
dst_root = Path(os.environ["PREP_QUALITY_CACHE_ROOT"])
allowlist_root = Path(os.environ["PREP_ALLOWLIST_ROOT"])
rounds = [r.strip() for r in os.environ["PREP_SIM_ROUNDS"].split(",") if r.strip()]

for round in rounds:
    dataset_name = f"synthetic_reaction_pdm_v1.0-{round}"
    src_cache = src_root / f"recogdrive_agent_cache_dir_{dataset_name}"
    dst_cache = dst_root / f"recogdrive_agent_cache_dir_{dataset_name}_quality"
    allowlist = allowlist_root / f"quality_filter_{dataset_name}" / "allowlist_log_token.tsv"
    if not src_cache.is_dir():
        raise RuntimeError(f"SimScale new-VLM source cache not found: {src_cache}")
    if not allowlist.is_file():
        raise RuntimeError(f"Quality allowlist not found: {allowlist}")

    linked = 0
    missing_src = 0
    dst_cache.mkdir(parents=True, exist_ok=True)
    with allowlist.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            log_name, token = line.split("\t", 1)
            src = src_cache / log_name / token
            if not src.is_dir():
                missing_src += 1
                continue
            dst_log = dst_cache / log_name
            dst_log.mkdir(parents=True, exist_ok=True)
            dst = dst_log / token
            if dst.exists() or dst.is_symlink():
                linked += 1
                continue
            try:
                dst.symlink_to(src, target_is_directory=True)
            except (FileExistsError, FileNotFoundError, OSError):
                if dst.exists() or dst.is_symlink():
                    linked += 1
                continue
            linked += 1

    if linked == 0:
        raise RuntimeError(f"No quality cache entries linked for round {round}: {dst_cache}")
    print(
        f"[fullmix-dit] round={round} linked_quality_cache={linked} "
        f"missing_src={missing_src} path={dst_cache}"
    )
PYPREP
elif [[ "${USE_QUALITY_CACHE}" == "true" ]]; then
  echo "[fullmix-dit] new-VLM quality caches already present; skip prep (set FORCE_PREP_QUALITY_CACHE=true to rebuild)"
fi

MIXED_CACHE_PATHS="${NAV_CACHE_PATH}"
MIXED_CACHE_NAMES="navtrain"
for idx in "${!SIM_CACHE_PATHS[@]}"; do
  MIXED_CACHE_PATHS="${MIXED_CACHE_PATHS},${SIM_CACHE_PATHS[$idx]}"
  MIXED_CACHE_NAMES="${MIXED_CACHE_NAMES},${SIM_CACHE_NAMES[$idx]}"
done

echo "[fullmix-dit] BASE_CKPT=${BASE_CKPT:-<none>}"
echo "[fullmix-dit] VLM_PATH=${VLM_PATH}"
echo "[fullmix-dit] NAV_CACHE_PATH=${NAV_CACHE_PATH}"
echo "[fullmix-dit] SIM_AGENT_CACHE_ROOT=${SIM_AGENT_CACHE_ROOT}"
echo "[fullmix-dit] QUALITY_CACHE_ROOT=${QUALITY_CACHE_ROOT}"
echo "[fullmix-dit] SIM_ROUNDS=${SIM_ROUNDS}"
echo "[fullmix-dit] USE_QUALITY_CACHE=${USE_QUALITY_CACHE}"
for idx in "${!SIM_CACHE_PATHS[@]}"; do
  echo "[fullmix-dit] SIM_CACHE_PATH[${SIM_CACHE_NAMES[$idx]}]=${SIM_CACHE_PATHS[$idx]}"
done
echo "[fullmix-dit] mixed_cache.fullmix=true (uniform union, sample_ratios ignored)"
echo "[fullmix-dit] CACHE_READ_MAX_RETRIES=${CACHE_READ_MAX_RETRIES} CACHE_READ_RETRY_BASE_SEC=${CACHE_READ_RETRY_BASE_SEC}"
echo "[fullmix-dit] CKPT_PATH=${CKPT_PATH:-<none>}"
echo "[fullmix-dit] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"

if [[ ! -d "${NAV_CACHE_PATH}" ]] || [[ -z "$(ls -A "${NAV_CACHE_PATH}" 2>/dev/null || true)" ]]; then
  echo "[ERROR] NAV_CACHE_PATH missing or empty: ${NAV_CACHE_PATH}" >&2
  exit 1
fi
for p in "${SIM_CACHE_PATHS[@]}"; do
  if [[ ! -d "${p}" ]] || [[ -z "$(ls -A "${p}" 2>/dev/null || true)" ]]; then
    echo "[ERROR] SIM cache missing or empty: ${p}" >&2
    exit 1
  fi
done
if [[ ! -d "${VLM_PATH}" ]]; then
  echo "[ERROR] VLM_PATH does not exist: ${VLM_PATH}" >&2
  exit 1
fi
if [[ -n "${CKPT_PATH}" && ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH does not exist: ${CKPT_PATH}" >&2
  exit 1
fi

HYDRA_ARGS=(
  agent=recogdrive_agent
  agent.lr="${LR}"
  agent.grpo=False
  agent.vlm_path="${VLM_PATH}"
  agent.cam_type='single'
  agent.cache_hidden_state=True
  agent.vlm_type='internvl'
  agent.dit_type='small'
  agent.vlm_size='small'
  agent.sampling_method='ddim'
  trainer.params.max_epochs="${MAX_EPOCHS}"
  trainer.params.num_nodes="${NNODES}"
  trainer.params.devices="${GPUS}"
  trainer.params.precision=bf16-mixed
  trainer.params.strategy=ddp_find_unused_parameters_true
  dataloader.params.batch_size="${BATCH_SIZE}"
  dataloader.params.num_workers="${NUM_WORKERS}"
  experiment_name="${EXPERIMENT_NAME}"
  train_test_split=navtrain
  cache_path="${NAV_CACHE_PATH}"
  use_cache_without_dataset=True
  force_cache_computation=False
  use_mixed_cache=True
  mixed_cache.fullmix=True
  "mixed_cache.paths=[${MIXED_CACHE_PATHS}]"
  "mixed_cache.names=[${MIXED_CACHE_NAMES}]"
  hydra/job_logging=stdout
  hydra.output_subdir=null
)
# Weight-only init only when not doing Lightning resume (avoid double-load).
# Quote paths for Hydra: checkpoint filenames contain '=' (e.g. epoch=15-step=...).
if [[ -n "${BASE_CKPT}" ]]; then
  HYDRA_ARGS+=("agent.checkpoint_path='${BASE_CKPT}'")
else
  HYDRA_ARGS+=("agent.checkpoint_path=null")
fi
if [[ -n "${CKPT_PATH}" ]]; then
  HYDRA_ARGS+=("ckpt_path='${CKPT_PATH}'")
fi

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  "${HYDRA_ARGS[@]}"

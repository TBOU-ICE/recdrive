#!/usr/bin/env bash
# Full-mix navtrain + SimScale DiT IL launcher.
#
# Unlike run_recogdrive_train_mixed_simscale_2b.sh (weighted ratio sampling),
# this script sets mixed_cache.fullmix=true so every cached sample has equal
# probability. The effective batch composition follows dataset size
# (navtrain ~N + simscale round0 ~M0 + simscale round1 ~M1), not a fixed ratio.
#
# By default both SimScale round0 and round1 quality-filtered caches are used.
# Override SIM_ROUNDS (comma-separated, e.g. "0" or "0,1") or SIMSCALE_ROOT as needed.
#
# Usage:
#   bash scripts/training/run_recogdrive_train_fullmix_simscale_2b.sh
#   GPUS=8 MAX_EPOCHS=20 bash scripts/training/run_recogdrive_train_fullmix_simscale_2b.sh
#   SIM_ROUNDS=0,1 SIMSCALE_ROOT=/workspace/datasets/simscale/20260709 bash scripts/training/run_recogdrive_train_fullmix_simscale_2b.sh

set -euo pipefail

export PATH="${CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/code}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
# Soften Alluxio/FUSE flaky reads in dataset.py load_feature_target_from_pickle.
export CACHE_READ_MAX_RETRIES="${CACHE_READ_MAX_RETRIES:-5}"
export CACHE_READ_RETRY_BASE_SEC="${CACHE_READ_RETRY_BASE_SEC:-0.5}"

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/datasets/simscale/20260709}"
SIM_ROUNDS="${SIM_ROUNDS:-0,1}"
USE_QUALITY_CACHE="${USE_QUALITY_CACHE:-true}"

NAV_CACHE_PATH="${NAV_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"

IFS=',' read -r -a SIM_ROUND_LIST <<< "${SIM_ROUNDS}"
SIM_CACHE_PATHS=()
SIM_CACHE_NAMES=()
for round in "${SIM_ROUND_LIST[@]}"; do
  round="${round//[[:space:]]/}"
  if [[ -z "${round}" ]]; then
    continue
  fi
  dataset_name="synthetic_reaction_pdm_v1.0-${round}"
  if [[ "${USE_QUALITY_CACHE}" == "true" ]]; then
    sim_cache_path="${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}_quality"
  else
    sim_cache_path="${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}"
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
VLM_PATH="${VLM_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23457}"
GPUS="${GPUS:-8}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_dit_il_fullmix_simscale_round01_quality}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/python}"
# Lightning full-state resume (model + optimizer + epoch/step). Continues from epoch 15.
CKPT_PATH="${CKPT_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_dit_il_fullmix_simscale_round01_quality/2026.07.12.03.07.49/lightning_logs/version_0/checkpoints/epoch=99-step=156100.ckpt}"

if [[ "${USE_QUALITY_CACHE}" == "true" ]]; then
  NEED_PREP_QUALITY_CACHE=false
  for round in "${SIM_ROUND_LIST[@]}"; do
    round="${round//[[:space:]]/}"
    [[ -z "${round}" ]] && continue
    quality_cache="${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-${round}_quality"
    if [[ ! -d "${quality_cache}" ]] || [[ -z "$(ls -A "${quality_cache}" 2>/dev/null || true)" ]]; then
      NEED_PREP_QUALITY_CACHE=true
      break
    fi
  done

  if [[ "${FORCE_PREP_QUALITY_CACHE:-false}" == "true" ]]; then
    NEED_PREP_QUALITY_CACHE=true
  fi

  if [[ "${NEED_PREP_QUALITY_CACHE}" == "true" ]]; then
  export PREP_SIMSCALE_ROOT="${SIMSCALE_ROOT}"
  export PREP_SIM_ROUNDS="${SIM_ROUNDS}"
  "${PYTHON_BIN}" - <<'PYPREP'
import os
from pathlib import Path

simscale_root = Path(os.environ["PREP_SIMSCALE_ROOT"])
rounds = [r.strip() for r in os.environ["PREP_SIM_ROUNDS"].split(",") if r.strip()]

for round in rounds:
    dataset_name = f"synthetic_reaction_pdm_v1.0-{round}"
    src_cache = simscale_root / f"recogdrive_agent_cache_dir_{dataset_name}"
    dst_cache = simscale_root / f"recogdrive_agent_cache_dir_{dataset_name}_quality"
    allowlist = simscale_root / f"quality_filter_{dataset_name}" / "allowlist_log_token.tsv"
    if not src_cache.is_dir():
        raise RuntimeError(f"SimScale source cache not found: {src_cache}")
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
  else
    echo "[fullmix-dit] quality symlink caches already present; skip prep (set FORCE_PREP_QUALITY_CACHE=true to rebuild)"
  fi
fi

MIXED_CACHE_PATHS="${NAV_CACHE_PATH}"
MIXED_CACHE_NAMES="navtrain"
for idx in "${!SIM_CACHE_PATHS[@]}"; do
  MIXED_CACHE_PATHS="${MIXED_CACHE_PATHS},${SIM_CACHE_PATHS[$idx]}"
  MIXED_CACHE_NAMES="${MIXED_CACHE_NAMES},${SIM_CACHE_NAMES[$idx]}"
done

echo "[fullmix-dit] BASE_CKPT=${BASE_CKPT:-<none>}"
echo "[fullmix-dit] NAV_CACHE_PATH=${NAV_CACHE_PATH}"
echo "[fullmix-dit] SIMSCALE_ROOT=${SIMSCALE_ROOT}"
echo "[fullmix-dit] SIM_ROUNDS=${SIM_ROUNDS}"
echo "[fullmix-dit] USE_QUALITY_CACHE=${USE_QUALITY_CACHE}"
for idx in "${!SIM_CACHE_PATHS[@]}"; do
  echo "[fullmix-dit] SIM_CACHE_PATH[${SIM_CACHE_NAMES[$idx]}]=${SIM_CACHE_PATHS[$idx]}"
done
echo "[fullmix-dit] mixed_cache.fullmix=true (uniform union, sample_ratios ignored)"
echo "[fullmix-dit] CACHE_READ_MAX_RETRIES=${CACHE_READ_MAX_RETRIES} CACHE_READ_RETRY_BASE_SEC=${CACHE_READ_RETRY_BASE_SEC}"
echo "[fullmix-dit] CKPT_PATH=${CKPT_PATH:-<none>}"
echo "[fullmix-dit] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"

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

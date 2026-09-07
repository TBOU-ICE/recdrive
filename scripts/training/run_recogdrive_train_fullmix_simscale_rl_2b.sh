#!/usr/bin/env bash
# Full-mix navtrain + SimScale GRPO/RL launcher.
#
# This is intentionally separate from the existing IL and RL launchers:
# - trains with run_training_recogdrive_mixed_simscale_rl.py;
# - keeps navtrain/simscale feature caches in their original locations;
# - writes only a small union metric-cache metadata CSV under MIX_ROOT;
# - uses PDM score (PDMS) as the GRPO reward;
# - does not add trajectory IL loss for SimScale samples.
#
# Usage:
#   bash scripts/training/run_recogdrive_train_fullmix_simscale_rl_2b.sh
#   GPUS=8 MASTER_PORT=23458 bash scripts/training/run_recogdrive_train_fullmix_simscale_rl_2b.sh
#   NAV_RATIO=0.9 SIM_RATIO=0.1 bash scripts/training/run_recogdrive_train_fullmix_simscale_rl_2b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PATH="${CONDA_BIN:-/opt/conda/envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-bd-su01/nby/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/datasets/recdrive/20260513/nby/recdrive/download}"
export PYTHONPATH="${REPO_ROOT}:${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

ROUND="${ROUND:-0}"
DATASET_NAME="${DATASET_NAME:-synthetic_reaction_pdm_v1.0-${ROUND}_quality}"
METRIC_DATASET_NAME="${METRIC_DATASET_NAME:-synthetic_reaction_pdm_v1.0-${ROUND}}"
SIMSCALE_ROOT="${SIMSCALE_ROOT:-/mnt/datasets/simscale/20260709}"

NAV_CACHE_PATH="${NAV_CACHE_PATH:-/mnt/models/recdrive/v1.0.0/recogdrive_agent_cache_dir_train}"
SIM_CACHE_PATH="${SIM_CACHE_PATH:-${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${DATASET_NAME}}"
NAV_METRIC_CACHE_PATH="${NAV_METRIC_CACHE_PATH:-/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache_train}"
SIM_METRIC_CACHE_PATH="${SIM_METRIC_CACHE_PATH:-${SIMSCALE_ROOT}/metric_cache_${METRIC_DATASET_NAME}}"
MIX_ROOT="${MIX_ROOT:-${SIMSCALE_ROOT}/mixed_training/fullmix_simscale_rl}"
MIX_METRIC_CACHE_PATH="${MIX_METRIC_CACHE_PATH:-${MIX_ROOT}/union_metric_cache}"

BASE_CKPT="${BASE_CKPT:-/mnt/models/recdrive/v1.0.0/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt}"
REF_CKPT="${REF_CKPT:-${BASE_CKPT}}"
VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/ReCogDrive-VLM-2B}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23458}"
GPUS="${GPUS:-8}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/recdrive/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/recdrive/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_rl_fullmix_simscale_quality_pdms}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"

NAV_RATIO="${NAV_RATIO:-}"
SIM_RATIO="${SIM_RATIO:-}"
MIX_NUM_SAMPLES="${MIX_NUM_SAMPLES:-0}"

mkdir -p "${MIX_METRIC_CACHE_PATH}/metadata"

export MIX_NAV_METRIC_CACHE_PATH="${NAV_METRIC_CACHE_PATH}"
export MIX_SIM_METRIC_CACHE_PATH="${SIM_METRIC_CACHE_PATH}"
export MIX_METRIC_CACHE_PATH="${MIX_METRIC_CACHE_PATH}"

"${PYTHON_BIN}" - <<'PYMERGE'
import csv
import os
from pathlib import Path

nav_metric = Path(os.environ["MIX_NAV_METRIC_CACHE_PATH"])
sim_metric = Path(os.environ["MIX_SIM_METRIC_CACHE_PATH"])
mix_metric = Path(os.environ["MIX_METRIC_CACHE_PATH"])

def collect_rows(metric_root: Path):
    metadata_dir = metric_root / "metadata"
    if not metadata_dir.is_dir():
        raise RuntimeError(f"Metric metadata dir does not exist: {metadata_dir}")
    rows = []
    for csv_path in sorted(metadata_dir.glob("*.csv")):
        if csv_path.name.endswith(".bak_oldpath"):
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file_name") or next(iter(row.values()), None)
                if file_name and Path(file_name).is_file():
                    rows.append(file_name)
    if not rows:
        raise RuntimeError(f"No metric cache rows found under {metadata_dir}")
    return rows

paths = []
seen = set()
for path in collect_rows(nav_metric) + collect_rows(sim_metric):
    token = Path(path).parts[-2] if len(Path(path).parts) >= 2 else path
    if token in seen:
        continue
    seen.add(token)
    paths.append(path)

out_csv = mix_metric / "metadata" / "mixed_metric_cache_metadata_node_0.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name"])
    for path in paths:
        writer.writerow([path])

print(f"[mixed-rl] Wrote {len(paths)} metric-cache rows to {out_csv}")
PYMERGE

echo "[mixed-rl] BASE_CKPT=${BASE_CKPT}"
echo "[mixed-rl] REF_CKPT=${REF_CKPT}"
echo "[mixed-rl] NAV_CACHE_PATH=${NAV_CACHE_PATH}"
echo "[mixed-rl] SIM_CACHE_PATH=${SIM_CACHE_PATH}"
echo "[mixed-rl] NAV_METRIC_CACHE_PATH=${NAV_METRIC_CACHE_PATH}"
echo "[mixed-rl] SIM_METRIC_CACHE_PATH=${SIM_METRIC_CACHE_PATH}"
echo "[mixed-rl] MIX_METRIC_CACHE_PATH=${MIX_METRIC_CACHE_PATH}"
echo "[mixed-rl] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[mixed-rl] reward: PDMS (PDM score)"
echo "[mixed-rl] SimScale trajectory IL loss is not added."

HYDRA_OVERRIDES=(
  agent=recogdrive_agent
  agent.lr="${LR}"
  agent.grpo=True
  agent.vlm_path="${VLM_PATH}"
  "agent.checkpoint_path=${BASE_CKPT}"
  "agent.reference_policy_checkpoint=${REF_CKPT}"
  agent.cam_type='single'
  agent.cache_hidden_state=True
  agent.vlm_type='internvl'
  agent.dit_type='small'
  agent.vlm_size='small'
  agent.sampling_method='ddim'
  agent.metric_cache_path="${MIX_METRIC_CACHE_PATH}"
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
  "mixed_cache.paths=[${NAV_CACHE_PATH},${SIM_CACHE_PATH}]"
  'mixed_cache.names=[navtrain,simscale]'
  hydra/job_logging=stdout
  hydra.output_subdir=null
)

if [[ -n "${NAV_RATIO}" || -n "${SIM_RATIO}" ]]; then
  NAV_RATIO="${NAV_RATIO:-0.9}"
  SIM_RATIO="${SIM_RATIO:-0.1}"
  echo "[mixed-rl] ratio sampling navtrain=${NAV_RATIO} simscale=${SIM_RATIO}"
  HYDRA_OVERRIDES+=(
    "mixed_cache.sample_ratios=[${NAV_RATIO},${SIM_RATIO}]"
    mixed_cache.num_samples="${MIX_NUM_SAMPLES}"
  )
else
  echo "[mixed-rl] fullmix=true (uniform union, sample_ratios ignored)"
  HYDRA_OVERRIDES+=(mixed_cache.fullmix=True)
fi

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS}" \
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive_mixed_simscale_rl.py" \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
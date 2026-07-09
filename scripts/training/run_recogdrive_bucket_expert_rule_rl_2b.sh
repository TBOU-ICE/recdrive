#!/usr/bin/env bash
set -euo pipefail

# Rule-intersection RL with three explicit sampling sources:
#   40% full navtrain cache
#   30% navtrain rule_intersection bucket
#   30% SimScale quality rule_intersection bucket
#
# The model is initialized from INIT_CKPT and the GRPO reference policy defaults
# to the same checkpoint.

export PATH="${CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/code}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/download}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS="${GPUS:-8}"

SIM_ROUND="${SIM_ROUND:-0}"
SIM_CACHE_DATASET_NAME="${SIM_CACHE_DATASET_NAME:-synthetic_reaction_pdm_v1.0-${SIM_ROUND}}"
SIM_DATASET_NAME="${SIM_DATASET_NAME:-synthetic_reaction_pdm_v1.0-${SIM_ROUND}_quality}"
SIM_METRIC_DATASET_NAME="${SIM_METRIC_DATASET_NAME:-synthetic_reaction_pdm_v1.0-${SIM_ROUND}}"

INIT_CKPT="${INIT_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_il_fullmix_simscale_quality/2026.07.07.03.49.53/lightning_logs/version_0/checkpoints/epoch=29-step=36480.ckpt}"
REF_CKPT="${REF_CKPT:-${INIT_CKPT}}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"
VLM_PATH="${VLM_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR:-/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain}"
SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/datasets/simscale/20260709}"
# scene_buckets metadata still lives on the old NFS root; cache/metric use SIMSCALE_ROOT.
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"
SIM_BUCKET_DIR="${SIM_BUCKET_DIR:-${SIMSCALE_BUCKET_ROOT}/scene_buckets_${SIM_DATASET_NAME}}"

NAV_CACHE_PATH="${NAV_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
SIM_CACHE_PATH="${SIM_CACHE_PATH:-${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${SIM_CACHE_DATASET_NAME}}"
NAV_METRIC_CACHE_PATH="${NAV_METRIC_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train}"
SIM_METRIC_CACHE_PATH="${SIM_METRIC_CACHE_PATH:-${SIMSCALE_ROOT}/metric_cache_${SIM_METRIC_DATASET_NAME}}"

MIX_ROOT="${MIX_ROOT:-${SIMSCALE_ROOT}/mixed_training/rule_intersection_45nav_35navbucket_20simbucket}"
NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH:-${MIX_ROOT}/navtrain_rule_bucket_cache}"
SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH:-${MIX_ROOT}/simscale_rule_bucket_cache}"
MIX_METRIC_CACHE_PATH="${MIX_METRIC_CACHE_PATH:-${MIX_ROOT}/union_metric_cache}"
MIX_INFO_DIR="${MIX_INFO_DIR:-${MIX_ROOT}/metadata}"

NAV_RATIO="${NAV_RATIO:-0.45}"
NAV_BUCKET_RATIO="${NAV_BUCKET_RATIO:-0.35}"
SIM_BUCKET_RATIO="${SIM_BUCKET_RATIO:-0.2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_rule_rl_40nav_30navbucket_30simbucket}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/python}"

mkdir -p "${NAV_BUCKET_CACHE_PATH}" "${SIM_BUCKET_CACHE_PATH}" "${MIX_METRIC_CACHE_PATH}/metadata" "${MIX_INFO_DIR}"

export MIX_NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR}"
export MIX_SIM_BUCKET_DIR="${SIM_BUCKET_DIR}"
export MIX_NAV_CACHE_PATH="${NAV_CACHE_PATH}"
export MIX_SIM_CACHE_PATH="${SIM_CACHE_PATH}"
export MIX_NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH}"
export MIX_SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH}"
export MIX_NAV_METRIC_CACHE_PATH="${NAV_METRIC_CACHE_PATH}"
export MIX_SIM_METRIC_CACHE_PATH="${SIM_METRIC_CACHE_PATH}"
export MIX_METRIC_CACHE_PATH="${MIX_METRIC_CACHE_PATH}"
export MIX_INFO_DIR="${MIX_INFO_DIR}"

"${PYTHON_BIN}" - <<'PYPREP'
import csv
import json
import os
from pathlib import Path

navtrain_output_dir = Path(os.environ["MIX_NAVTRAIN_OUTPUT_DIR"])
sim_bucket_dir = Path(os.environ["MIX_SIM_BUCKET_DIR"])
nav_cache = Path(os.environ["MIX_NAV_CACHE_PATH"])
sim_cache = Path(os.environ["MIX_SIM_CACHE_PATH"])
nav_bucket_cache = Path(os.environ["MIX_NAV_BUCKET_CACHE_PATH"])
sim_bucket_cache = Path(os.environ["MIX_SIM_BUCKET_CACHE_PATH"])
nav_metric = Path(os.environ["MIX_NAV_METRIC_CACHE_PATH"])
sim_metric = Path(os.environ["MIX_SIM_METRIC_CACHE_PATH"])
mix_metric = Path(os.environ["MIX_METRIC_CACHE_PATH"])
info_dir = Path(os.environ["MIX_INFO_DIR"])

bucket_file = "exclusive_rule_intersection_tokens.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_token(token):
    token = str(token).strip().lower()
    base, sep, suffix = token.rpartition("-")
    if sep and suffix.isdigit() and len(suffix) == 3 and base:
        return token
    return token.replace("-", "")


def load_token_to_log(path):
    data = load_json(path)
    out = {}
    for token, meta in data.items():
        if not isinstance(meta, dict):
            continue
        log_name = meta.get("log_name")
        norm = normalize_token(token)
        if norm and log_name:
            out[norm] = log_name
    return out


def link_token(src_root, dst_root, log_name, token):
    src = src_root / log_name / token
    if not src.is_dir():
        return False
    dst_log = dst_root / log_name
    dst_log.mkdir(parents=True, exist_ok=True)
    dst = dst_log / token
    if dst.exists() or dst.is_symlink():
        return True
    dst.symlink_to(src, target_is_directory=True)
    return True


def iter_metric_paths(metric_root):
    metadata_dir = metric_root / "metadata"
    if not metadata_dir.is_dir():
        return []
    paths = []
    for csv_path in sorted(metadata_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get("file_name") or next(iter(row.values()))
                if path:
                    paths.append(path)
    return paths


nav_token_to_log = load_token_to_log(navtrain_output_dir / "navtrain_token_to_buckets.json")
sim_token_to_log = load_token_to_log(sim_bucket_dir / "simscale_token_to_buckets.json")

nav_rule_tokens = [normalize_token(t) for t in load_json(navtrain_output_dir / bucket_file)]
nav_rule_tokens = [t for t in nav_rule_tokens if t in nav_token_to_log]

sim_rule_tokens = [normalize_token(t) for t in load_json(sim_bucket_dir / bucket_file)]
sim_rule_tokens = [t for t in sim_rule_tokens if t in sim_token_to_log]

linked_nav_bucket = 0
for token in nav_rule_tokens:
    linked_nav_bucket += int(link_token(nav_cache, nav_bucket_cache, nav_token_to_log[token], token))

linked_sim_bucket = 0
for token in sim_rule_tokens:
    linked_sim_bucket += int(link_token(sim_cache, sim_bucket_cache, sim_token_to_log[token], token))

if linked_nav_bucket == 0:
    raise RuntimeError("No navtrain rule bucket cache entries were linked.")
if linked_sim_bucket == 0:
    raise RuntimeError("No SimScale rule bucket cache entries were linked.")

sim_rule_set = set(sim_rule_tokens)
metric_csv = mix_metric / "metadata" / "mixed_metric_cache_metadata_node_0.csv"
metric_csv.parent.mkdir(parents=True, exist_ok=True)
with metric_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name"])
    for path in iter_metric_paths(nav_metric):
        writer.writerow([path])
    for path in iter_metric_paths(sim_metric):
        token = normalize_token(Path(path).parts[-2])
        if token in sim_rule_set:
            writer.writerow([path])

summary = {
    "nav_rule_tokens": len(nav_rule_tokens),
    "sim_rule_tokens": len(sim_rule_tokens),
    "linked_nav_bucket": linked_nav_bucket,
    "linked_sim_bucket": linked_sim_bucket,
    "nav_bucket_cache": str(nav_bucket_cache),
    "sim_bucket_cache": str(sim_bucket_cache),
    "mixed_metric_cache": str(mix_metric),
}
save_json(summary, info_dir / "rule_mixed_sources_summary.json")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PYPREP

echo "INIT_CKPT: ${INIT_CKPT}"
echo "REF_CKPT: ${REF_CKPT}"
echo "NAV_CACHE_PATH: ${NAV_CACHE_PATH}"
echo "NAV_BUCKET_CACHE_PATH: ${NAV_BUCKET_CACHE_PATH}"
echo "SIM_BUCKET_CACHE_PATH: ${SIM_BUCKET_CACHE_PATH}"
echo "MIX_METRIC_CACHE_PATH: ${MIX_METRIC_CACHE_PATH}"
echo "SAMPLE_RATIOS: nav=${NAV_RATIO}, nav_rule=${NAV_BUCKET_RATIO}, sim_rule=${SIM_BUCKET_RATIO}"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_mixed_simscale_rl.py" \
  agent=recogdrive_agent \
  "agent.checkpoint_path=${INIT_CKPT}" \
  "agent.reference_policy_checkpoint=${REF_CKPT}" \
  agent.lr="${LR}" \
  agent.grpo=True \
  agent.vlm_path="${VLM_PATH}" \
  agent.cam_type='single' \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.dit_type='small' \
  agent.vlm_size='small' \
  agent.sampling_method='ddim' \
  agent.metric_cache_path="${MIX_METRIC_CACHE_PATH}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.params.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  cache_path="${NAV_CACHE_PATH}" \
  use_cache_without_dataset=true \
  use_mixed_cache=true \
  mixed_cache.paths="[${NAV_CACHE_PATH},${NAV_BUCKET_CACHE_PATH},${SIM_BUCKET_CACHE_PATH}]" \
  mixed_cache.names="[navtrain_full,navtrain_rule,simscale_rule]" \
  mixed_cache.sample_ratios="[${NAV_RATIO},${NAV_BUCKET_RATIO},${SIM_BUCKET_RATIO}]" \
  mixed_cache.fullmix=false \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# STANDALONE teacher trainer: T3 - progress_curbside_stopgo.
# Self-contained; run on ONE 8-GPU machine (trains only this teacher):
#   GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_progress_rl_newvlm.sh
#
# Reward (REWARD_MODE=progress): gate(NC,DAC)*(0.6*PDMS + 0.4*EP).
# Three weighted sampling sources: full navtrain / navtrain-progress / SimScale-progress.
# Guardrail: watch NC/DAC do not collapse while chasing progress.
#
# Representation contract (MUST stay consistent, else the DiT reads OOD features):
#   VLM_PATH must match the VLM that produced the agent caches AND INIT_CKPT
#   (default: vlm_simscale_lora_merged, LLM-only LoRA). The '..._vit_...' cache is a
#   DIFFERENT representation - do not use it here.
# =============================================================================

# ---- repo root resolved from THIS script's location (portable across machines) ----
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

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23473}"
GPUS="${GPUS:-8}"

# ---- teacher identity (baked in for this standalone script) ----
BUCKET_NAME="${BUCKET_NAME:-progress_curbside_stopgo}"
BUCKET_FILE="${BUCKET_FILE:-exclusive_progress_curbside_stopgo_tokens.json}"
export REWARD_MODE="${REWARD_MODE:-progress}"

# ---- sampling ratios (full navtrain / navtrain-bucket / simscale-bucket) ----
NAV_RATIO="${NAV_RATIO:-0.0}"
NAV_BUCKET_RATIO="${NAV_BUCKET_RATIO:-0.5}"
SIM_BUCKET_RATIO="${SIM_BUCKET_RATIO:-0.5}"

# ---- new-representation init + VLM (must match the caches!) ----
INIT_CKPT="${INIT_CKPT:-/mnt/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.20.14.31.06/lightning_logs/version_0/checkpoints/epoch=199-step=312200.ckpt}"
REF_CKPT="${REF_CKPT:-${INIT_CKPT}}"
VLM_PATH="${VLM_PATH:-/mnt/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"

# Optional full-state resume (continue optimizer+epoch from a prior RL run of THIS teacher).
# Empty -> fresh run warm-started from INIT_CKPT (stage-1 IL). To continue a run, point at that
# teacher's checkpoint, e.g. RESUME_CKPT=<exp>/.../lightning_logs/version_0/checkpoints/last.ckpt
RESUME_CKPT="${RESUME_CKPT:-}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[teacher] RESUME full state from: ${RESUME_CKPT}"
  CKPT_ARGS=( "agent.checkpoint_path=null" "ckpt_path='${RESUME_CKPT}'" )
else
  echo "[teacher] FRESH warm-start from INIT_CKPT: ${INIT_CKPT}"
  CKPT_ARGS=( "agent.checkpoint_path='${INIT_CKPT}'" )
fi

LR="${LR:-3e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

# ---- data roots ----
# NEW-VLM (LLM-only LoRA = vlm_simscale_lora_merged) agent hidden-state caches.
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/mnt/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"
# Bucket token lists (quality-filtered, representation-INDEPENDENT). CPFS copy.
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/mnt/datasets/simscale/20260709/data/simscale}"
NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR:-/mnt/datasets/simscale/20260709/data/navtrain_scene/output/navtrain}"
# Metric caches (GRPO reward, representation-INDEPENDENT). Keep on datasets (.pkl live there).
SIMSCALE_METRIC_ROOT="${SIMSCALE_METRIC_ROOT:-/mnt/datasets/simscale/20260709}"
NAV_METRIC_CACHE_PATH="${NAV_METRIC_CACHE_PATH:-/mnt/datasets/recdrive/20260513/nby/recdrive/metric_cache_train}"

SIM_ROUNDS="${SIM_ROUNDS:-${SIM_ROUND:-0,1}}"

IFS=',' read -r -a SIM_ROUND_LIST <<< "${SIM_ROUNDS}"
SIM_ROUNDS_CLEAN=()
SIM_CACHE_PATHS=()
SIM_BUCKET_DIRS=()
SIM_METRIC_CACHE_PATHS=()
for round in "${SIM_ROUND_LIST[@]}"; do
  round="${round//[[:space:]]/}"
  if [[ -z "${round}" ]]; then
    continue
  fi
  SIM_ROUNDS_CLEAN+=("${round}")
  dataset_name="synthetic_reaction_pdm_v1.0-${round}"
  quality_cache="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}_quality"
  full_cache="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}"
  if [[ -d "${quality_cache}" ]]; then
    SIM_CACHE_PATHS+=("${quality_cache}")
  else
    SIM_CACHE_PATHS+=("${full_cache}")
  fi
  SIM_BUCKET_DIRS+=("${SIMSCALE_BUCKET_ROOT}/scene_buckets_${dataset_name}_quality")
  SIM_METRIC_CACHE_PATHS+=("${SIMSCALE_METRIC_ROOT}/metric_cache_${dataset_name}")
done

if [[ ${#SIM_CACHE_PATHS[@]} -eq 0 ]]; then
  echo "[ERROR] SIM_ROUNDS resolved to no cache paths: ${SIM_ROUNDS}" >&2
  exit 1
fi
SIM_ROUNDS="${SIM_ROUNDS_CLEAN[*]}"
SIM_ROUNDS="${SIM_ROUNDS// /,}"

# Per-teacher mixed-training workspace (isolated so teachers never share symlink caches).
MIX_ROOT="${MIX_ROOT:-/mnt/datasets/simscale/20260709/data/simscale/mixed_training_newvlm/${BUCKET_NAME}_${NAV_RATIO}nav_${NAV_BUCKET_RATIO}navbucket_${SIM_BUCKET_RATIO}simbucket}"
NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH:-${MIX_ROOT}/navtrain_bucket_cache}"
SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH:-${MIX_ROOT}/simscale_bucket_cache}"
MIX_METRIC_CACHE_PATH="${MIX_METRIC_CACHE_PATH:-${MIX_ROOT}/union_metric_cache}"
MIX_INFO_DIR="${MIX_INFO_DIR:-${MIX_ROOT}/metadata}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_teacher_${BUCKET_NAME}_rl_newvlm_v1-1}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/recdrive/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/recdrive/bin/python}"

mkdir -p "${NAV_BUCKET_CACHE_PATH}" "${SIM_BUCKET_CACHE_PATH}" "${MIX_METRIC_CACHE_PATH}/metadata" "${MIX_INFO_DIR}"

export MIX_NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR}"
export MIX_NAV_CACHE_PATH="${NAV_CACHE_PATH}"
export MIX_NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH}"
export MIX_SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH}"
export MIX_NAV_METRIC_CACHE_PATH="${NAV_METRIC_CACHE_PATH}"
export MIX_METRIC_CACHE_PATH="${MIX_METRIC_CACHE_PATH}"
export MIX_INFO_DIR="${MIX_INFO_DIR}"
export MIX_BUCKET_FILE="${BUCKET_FILE}"

IFS=','; export MIX_SIM_CACHE_PATHS="${SIM_CACHE_PATHS[*]}"; unset IFS
IFS=','; export MIX_SIM_BUCKET_DIRS="${SIM_BUCKET_DIRS[*]}"; unset IFS
IFS=','; export MIX_SIM_METRIC_CACHE_PATHS="${SIM_METRIC_CACHE_PATHS[*]}"; unset IFS

"${PYTHON_BIN}" - <<'PYPREP'
import csv
import json
import os
from pathlib import Path

navtrain_output_dir = Path(os.environ["MIX_NAVTRAIN_OUTPUT_DIR"])
sim_bucket_dirs = [Path(p) for p in os.environ["MIX_SIM_BUCKET_DIRS"].split(",") if p.strip()]
nav_cache = Path(os.environ["MIX_NAV_CACHE_PATH"])
sim_caches = [Path(p) for p in os.environ["MIX_SIM_CACHE_PATHS"].split(",") if p.strip()]
nav_bucket_cache = Path(os.environ["MIX_NAV_BUCKET_CACHE_PATH"])
sim_bucket_cache = Path(os.environ["MIX_SIM_BUCKET_CACHE_PATH"])
nav_metric = Path(os.environ["MIX_NAV_METRIC_CACHE_PATH"])
sim_metrics = [Path(p) for p in os.environ["MIX_SIM_METRIC_CACHE_PATHS"].split(",") if p.strip()]
mix_metric = Path(os.environ["MIX_METRIC_CACHE_PATH"])
info_dir = Path(os.environ["MIX_INFO_DIR"])
bucket_file = os.environ.get("MIX_BUCKET_FILE", "exclusive_rule_intersection_tokens.json")

if not (len(sim_bucket_dirs) == len(sim_caches) == len(sim_metrics)):
    raise RuntimeError(
        "SimScale round lists must have equal length: "
        f"buckets={len(sim_bucket_dirs)} caches={len(sim_caches)} metrics={len(sim_metrics)}"
    )


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
        if csv_path.name.endswith(".bak_oldpath"):
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get("file_name") or next(iter(row.values()))
                if path and Path(path).is_file():
                    paths.append(path)
    return paths


nav_token_to_log = load_token_to_log(navtrain_output_dir / "navtrain_token_to_buckets.json")
nav_bucket_tokens = [normalize_token(t) for t in load_json(navtrain_output_dir / bucket_file)]
nav_bucket_tokens = [t for t in nav_bucket_tokens if t in nav_token_to_log]

linked_nav_bucket = 0
for token in nav_bucket_tokens:
    linked_nav_bucket += int(link_token(nav_cache, nav_bucket_cache, nav_token_to_log[token], token))

if linked_nav_bucket == 0:
    raise RuntimeError(
        f"No navtrain bucket cache entries were linked for {bucket_file}. "
        f"Check NAV_CACHE_PATH={nav_cache} matches the new-VLM representation."
    )

sim_bucket_tokens = []
linked_sim_bucket = 0
linked_sim_by_round = {}
for sim_bucket_dir, sim_cache in zip(sim_bucket_dirs, sim_caches):
    token_map_path = sim_bucket_dir / "simscale_token_to_buckets.json"
    rule_path = sim_bucket_dir / bucket_file
    if not token_map_path.is_file():
        raise RuntimeError(f"Missing SimScale token map: {token_map_path}")
    if not rule_path.is_file():
        raise RuntimeError(f"Missing SimScale bucket file: {rule_path}")
    if not sim_cache.is_dir():
        raise RuntimeError(f"Missing SimScale agent cache: {sim_cache}")

    sim_token_to_log = load_token_to_log(token_map_path)
    round_tokens = [normalize_token(t) for t in load_json(rule_path)]
    round_tokens = [t for t in round_tokens if t in sim_token_to_log]
    sim_bucket_tokens.extend(round_tokens)

    linked_round = 0
    for token in round_tokens:
        linked_round += int(link_token(sim_cache, sim_bucket_cache, sim_token_to_log[token], token))
    linked_sim_bucket += linked_round
    linked_sim_by_round[str(sim_bucket_dir)] = {
        "bucket_tokens": len(round_tokens),
        "linked": linked_round,
        "agent_cache": str(sim_cache),
    }

if linked_sim_bucket == 0:
    raise RuntimeError(
        "No SimScale bucket cache entries were linked. "
        "Check SIM_AGENT_CACHE_ROOT matches the new-VLM representation."
    )

sim_bucket_set = set(sim_bucket_tokens)
metric_csv = mix_metric / "metadata" / "mixed_metric_cache_metadata_node_0.csv"
metric_csv.parent.mkdir(parents=True, exist_ok=True)
with metric_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name"])
    for path in iter_metric_paths(nav_metric):
        writer.writerow([path])
    for sim_metric in sim_metrics:
        for path in iter_metric_paths(sim_metric):
            token = normalize_token(Path(path).parts[-2])
            if token in sim_bucket_set:
                writer.writerow([path])

summary = {
    "bucket_file": bucket_file,
    "nav_bucket_tokens": len(nav_bucket_tokens),
    "sim_bucket_tokens": len(sim_bucket_tokens),
    "sim_bucket_tokens_unique": len(sim_bucket_set),
    "linked_nav_bucket": linked_nav_bucket,
    "linked_sim_bucket": linked_sim_bucket,
    "linked_sim_by_round": linked_sim_by_round,
    "nav_bucket_cache": str(nav_bucket_cache),
    "sim_bucket_cache": str(sim_bucket_cache),
    "mixed_metric_cache": str(mix_metric),
}
save_json(summary, info_dir / "bucket_mixed_sources_summary.json")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PYPREP

echo "======================================================================"
echo "[teacher] BUCKET_NAME=${BUCKET_NAME}  BUCKET_FILE=${BUCKET_FILE}"
echo "[teacher] REWARD_MODE=${REWARD_MODE}"
echo "[teacher] SAMPLE_RATIOS: nav=${NAV_RATIO}, nav_bucket=${NAV_BUCKET_RATIO}, sim_bucket=${SIM_BUCKET_RATIO}"
echo "[teacher] INIT_CKPT=${INIT_CKPT}"
echo "[teacher] REF_CKPT=${REF_CKPT}"
echo "[teacher] VLM_PATH=${VLM_PATH}"
echo "[teacher] NAV_CACHE_PATH=${NAV_CACHE_PATH}"
echo "[teacher] SIM_AGENT_CACHE_ROOT=${SIM_AGENT_CACHE_ROOT}  SIM_ROUNDS=${SIM_ROUNDS}"
echo "[teacher] MIX_ROOT=${MIX_ROOT}"
echo "[teacher] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[teacher] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "======================================================================"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_mixed_simscale_rl.py" \
  agent=recogdrive_agent \
  "${CKPT_ARGS[@]}" \
  "agent.reference_policy_checkpoint=\"${REF_CKPT}\"" \
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
  force_cache_computation=False \
  use_mixed_cache=true \
  mixed_cache.paths="[${NAV_CACHE_PATH},${NAV_BUCKET_CACHE_PATH},${SIM_BUCKET_CACHE_PATH}]" \
  mixed_cache.names="[navtrain_full,navtrain_bucket,simscale_bucket]" \
  mixed_cache.sample_ratios="[${NAV_RATIO},${NAV_BUCKET_RATIO},${SIM_BUCKET_RATIO}]" \
  mixed_cache.fullmix=false \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

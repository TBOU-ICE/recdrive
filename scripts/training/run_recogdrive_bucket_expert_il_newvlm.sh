#!/usr/bin/env bash
set -euo pipefail

# Shared no-goal bucket IL teacher. The four wrappers set BUCKET_NAME / BUCKET_FILE
# and exec this file. Do not run this directly unless those two are exported.
#
# Mixture: fullmix union of THIS bucket's navtrain + SimScale quality tokens.
# Agent: recogdrive_agent (no privileged goal).
# Init: weight-only load from CKPT_PATH (default: new-VLM fullmix IL epoch=2).
# Optional RESUME_CKPT: Lightning full-state resume of THIS teacher.

if [[ -z "${BUCKET_NAME:-}" || -z "${BUCKET_FILE:-}" ]]; then
  echo "[ERROR] BUCKET_NAME and BUCKET_FILE must be set." >&2
  echo "  Use one of:" >&2
  echo "    bash scripts/training/run_recogdrive_bucket_expert_rule_il_newvlm.sh" >&2
  echo "    bash scripts/training/run_recogdrive_bucket_expert_safety_il_newvlm.sh" >&2
  echo "    bash scripts/training/run_recogdrive_bucket_expert_progress_il_newvlm.sh" >&2
  echo "    bash scripts/training/run_recogdrive_bucket_expert_general_il_newvlm.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"

export PATH="${CONDA_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CACHE_READ_MAX_RETRIES="${CACHE_READ_MAX_RETRIES:-10}"
export CACHE_READ_RETRY_BASE_SEC="${CACHE_READ_RETRY_BASE_SEC:-0.5}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23670}"
GPUS="${GPUS:-8}"

VLM_PATH="${VLM_PATH:-/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged}"
# Weight-only warm start (agent.initialize). Default: new-VLM fullmix IL epoch=2.
CKPT_PATH="${CKPT_PATH:-/workspace/models/recdrive/v1.0.0/training_dit_il_fullmix_simscale_newvlm/2026.07.27.11.45.27/lightning_logs/version_0/checkpoints/ckpt/epoch=2-step=4683.ckpt}"
# Optional Lightning full-state resume (optimizer + epoch) of a prior run of THIS teacher.
RESUME_CKPT="${RESUME_CKPT:-}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[teacher-il] RESUME full state from: ${RESUME_CKPT}"
  CKPT_ARGS=( "agent.checkpoint_path=null" "ckpt_path='${RESUME_CKPT}'" )
elif [[ -n "${CKPT_PATH}" ]]; then
  echo "[teacher-il] FRESH warm-start from CKPT_PATH: ${CKPT_PATH}"
  CKPT_ARGS=( "agent.checkpoint_path='${CKPT_PATH}'" )
else
  echo "[teacher-il] FRESH training from a RANDOM-INIT DiT"
  CKPT_ARGS=( "agent.checkpoint_path=null" )
fi

LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/workspace/datasets/simscale/20260709/data/simscale/new_vlm_quality_views}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/datasets/simscale/20260709/data/simscale}"
NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR:-/workspace/datasets/simscale/20260709/data/navtrain_scene/output/navtrain}"

SIM_ROUNDS="${SIM_ROUNDS:-${SIM_ROUND:-0,1}}"

IFS=',' read -r -a SIM_ROUND_LIST <<< "${SIM_ROUNDS}"
SIM_ROUNDS_CLEAN=()
SIM_CACHE_PATHS=()
SIM_BUCKET_DIRS=()
for round in "${SIM_ROUND_LIST[@]}"; do
  round="${round//[[:space:]]/}"
  if [[ -z "${round}" ]]; then
    continue
  fi
  SIM_ROUNDS_CLEAN+=("${round}")
  dataset_name="synthetic_reaction_pdm_v1.0-${round}"
  quality_cache="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}_quality"
  quality_cache_cpfs="${SIM_QUALITY_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}_quality"
  full_cache="${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}"
  if [[ -d "${quality_cache}" ]]; then
    SIM_CACHE_PATHS+=("${quality_cache}")
  elif [[ -d "${quality_cache_cpfs}" ]]; then
    SIM_CACHE_PATHS+=("${quality_cache_cpfs}")
  else
    SIM_CACHE_PATHS+=("${full_cache}")
  fi
  SIM_BUCKET_DIRS+=("${SIMSCALE_BUCKET_ROOT}/scene_buckets_${dataset_name}_quality")
done

if [[ ${#SIM_CACHE_PATHS[@]} -eq 0 ]]; then
  echo "[ERROR] SIM_ROUNDS resolved to no cache paths: ${SIM_ROUNDS}" >&2
  exit 1
fi
SIM_ROUNDS="${SIM_ROUNDS_CLEAN[*]}"
SIM_ROUNDS="${SIM_ROUNDS// /,}"

MIX_ROOT="${MIX_ROOT:-/workspace/datasets/simscale/20260709/data/simscale/il_training_newvlm/${BUCKET_NAME}_fullmix}"
NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH:-${MIX_ROOT}/navtrain_bucket_cache}"
SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH:-${MIX_ROOT}/simscale_bucket_cache}"
MIX_INFO_DIR="${MIX_INFO_DIR:-${MIX_ROOT}/metadata}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_teacher_2epoch_base_${BUCKET_NAME}_il_newvlm}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/python}"

mkdir -p "${NAV_BUCKET_CACHE_PATH}" "${SIM_BUCKET_CACHE_PATH}" "${MIX_INFO_DIR}"

if [[ ! -d "${NAV_CACHE_PATH}" ]] || [[ -z "$(ls -A "${NAV_CACHE_PATH}" 2>/dev/null || true)" ]]; then
  echo "[ERROR] NAV_CACHE_PATH missing or empty: ${NAV_CACHE_PATH}" >&2
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
if [[ -z "${RESUME_CKPT}" && -n "${CKPT_PATH}" && ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH does not exist: ${CKPT_PATH}" >&2
  exit 1
fi

export MIX_NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR}"
export MIX_NAV_CACHE_PATH="${NAV_CACHE_PATH}"
export MIX_NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH}"
export MIX_SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH}"
export MIX_INFO_DIR="${MIX_INFO_DIR}"
export MIX_BUCKET_FILE="${BUCKET_FILE}"

IFS=','; export MIX_SIM_CACHE_PATHS="${SIM_CACHE_PATHS[*]}"; unset IFS
IFS=','; export MIX_SIM_BUCKET_DIRS="${SIM_BUCKET_DIRS[*]}"; unset IFS

"${PYTHON_BIN}" - <<'PYPREP'
import json
import os
from pathlib import Path

navtrain_output_dir = Path(os.environ["MIX_NAVTRAIN_OUTPUT_DIR"])
sim_bucket_dirs = [Path(p) for p in os.environ["MIX_SIM_BUCKET_DIRS"].split(",") if p.strip()]
nav_cache = Path(os.environ["MIX_NAV_CACHE_PATH"])
sim_caches = [Path(p) for p in os.environ["MIX_SIM_CACHE_PATHS"].split(",") if p.strip()]
nav_bucket_cache = Path(os.environ["MIX_NAV_BUCKET_CACHE_PATH"])
sim_bucket_cache = Path(os.environ["MIX_SIM_BUCKET_CACHE_PATH"])
info_dir = Path(os.environ["MIX_INFO_DIR"])
bucket_file = os.environ["MIX_BUCKET_FILE"]

if len(sim_bucket_dirs) != len(sim_caches):
    raise RuntimeError(
        "SimScale round lists must have equal length: "
        f"buckets={len(sim_bucket_dirs)} caches={len(sim_caches)}"
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

summary = {
    "stage": "il",
    "bucket_file": bucket_file,
    "nav_bucket_tokens": len(nav_bucket_tokens),
    "sim_bucket_tokens": len(sim_bucket_tokens),
    "sim_bucket_tokens_unique": len(set(sim_bucket_tokens)),
    "linked_nav_bucket": linked_nav_bucket,
    "linked_sim_bucket": linked_sim_bucket,
    "linked_sim_by_round": linked_sim_by_round,
    "nav_bucket_cache": str(nav_bucket_cache),
    "sim_bucket_cache": str(sim_bucket_cache),
}
save_json(summary, info_dir / "bucket_il_sources_summary.json")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PYPREP

echo "======================================================================"
echo "[teacher-il] BUCKET_NAME=${BUCKET_NAME}  BUCKET_FILE=${BUCKET_FILE}"
echo "[teacher-il] GOAL: none (plain recogdrive_agent)"
echo "[teacher-il] MIXTURE: full-mix union of navtrain_bucket + simscale_bucket (no ratios)"
echo "[teacher-il] CKPT_PATH=${CKPT_PATH:-<none>}"
echo "[teacher-il] RESUME_CKPT=${RESUME_CKPT:-<none>}"
echo "[teacher-il] VLM_PATH=${VLM_PATH}"
echo "[teacher-il] NAV_CACHE_PATH=${NAV_CACHE_PATH}"
echo "[teacher-il] SIM_AGENT_CACHE_ROOT=${SIM_AGENT_CACHE_ROOT}"
echo "[teacher-il] SIM_QUALITY_CACHE_ROOT=${SIM_QUALITY_CACHE_ROOT}  SIM_ROUNDS=${SIM_ROUNDS}"
for _p in "${SIM_CACHE_PATHS[@]}"; do echo "[teacher-il] SIM_CACHE=${_p}"; done
echo "[teacher-il] MIX_ROOT=${MIX_ROOT}"
echo "[teacher-il] LR=${LR}  MAX_EPOCHS=${MAX_EPOCHS}  BATCH_SIZE=${BATCH_SIZE}"
echo "[teacher-il] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[teacher-il] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "======================================================================"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  agent=recogdrive_agent \
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
  hydra/job_logging=stdout \
  hydra.output_subdir=null

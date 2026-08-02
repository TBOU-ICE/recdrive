#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# STANDALONE teacher trainer (STAGE-1 IL, GOAL-CONDITIONED): T4 - general_or_no_tag.
# Self-contained; run on ONE 8-GPU machine (trains only this teacher):
#   GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_general_il_goal_newvlm.sh
#
# This is the imitation-learning counterpart of
# run_recogdrive_bucket_expert_general_rl_newvlm.sh. Same bucket and the same representation
# contract, but the data mixture is the plain union of the two bucket caches
# (no sampling ratios) -- the differences are:
#   * run_training_recogdrive.py instead of ..._mixed_simscale_rl.py
#   * agent=recogdrive_goal_agent with agent.goal_mode -> the PRIVILEGED goal point
#     (the ground-truth 8th waypoint) conditions the diffusion planner
#   * agent.grpo=False, so no reward, no reference policy, no metric cache
#   * random-init DiT by default (BASE_CKPT empty), 200 epochs at lr 1e-4
#
# The goal point is privileged information: it is read from
# targets["trajectory"][:, -1, :] and is available ONLY during training and during
# oracle evaluation. The student distilled from these teachers never sees it.
#
# GOAL_MODE selects where the goal is injected (see navsim/agents/recogdrive/goal_cond.py):
#   adaln   - added to the AdaLN conditioning vector, next to the ego status
#   channel - added into the residual stream next to the fused DiT input
#   cross   - appended as one extra cross-attention key/value token
#   none    - no goal, i.e. a plain IL baseline for the control group
#
# Representation contract (MUST stay consistent, else the DiT reads OOD features):
#   VLM_PATH must match the VLM that produced the agent caches
#   (default: vlm_simscale_lora_merged, LLM-only LoRA). The '..._vit_...' cache is a
#   DIFFERENT representation - do not use it here. Keeping this identical to the RL
#   script is what lets the stage-2 RL run warm-start from this checkpoint.
# =============================================================================

# ---- repo root resolved from THIS script's location (portable across machines) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"

export PATH="${CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23574}"
GPUS="${GPUS:-8}"

# ---- teacher identity (baked in for this standalone script) ----
BUCKET_NAME="${BUCKET_NAME:-general_or_no_tag}"
BUCKET_FILE="${BUCKET_FILE:-exclusive_general_or_no_tag_tokens.json}"

# ---- goal-point conditioning ----
GOAL_MODE="${GOAL_MODE:-adaln}"

# ---- data mixture: NO sampling ratios ----
# mixed_cache.fullmix=true makes the trainer shuffle the UNION of this bucket's
# navtrain and SimScale caches with uniform per-sample probability, i.e. the two
# scene sources contribute in their natural proportion and no WeightedRandomSampler
# is built. The full (un-bucketed) navtrain cache stays out of the training mixture;
# it is still read via cache_path for validation.

# ---- VLM (must match the caches!) ----
VLM_PATH="${VLM_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/vlm_simscale_lora_merged}"

# Weight-only warm start (agent.initialize). Empty = random-init DiT, which is the
# intended default for stage-1 IL. The goal branches are zero-initialised, so a
# non-empty BASE_CKPT reproduces that checkpoint exactly before the first step.
BASE_CKPT="${BASE_CKPT:-}"
# Optional full-state resume (optimizer + epoch) of a prior IL run of THIS teacher.
RESUME_CKPT="${RESUME_CKPT:-}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[teacher-il] RESUME full state from: ${RESUME_CKPT}"
  CKPT_ARGS=( "agent.checkpoint_path=null" "ckpt_path='${RESUME_CKPT}'" )
elif [[ -n "${BASE_CKPT}" ]]; then
  echo "[teacher-il] FRESH warm-start from BASE_CKPT: ${BASE_CKPT}"
  CKPT_ARGS=( "agent.checkpoint_path='${BASE_CKPT}'" )
else
  echo "[teacher-il] FRESH training from a RANDOM-INIT DiT"
  CKPT_ARGS=( "agent.checkpoint_path=null" )
fi

# IL hyperparameters. MAX_EPOCHS=200 is deliberate: ReCogDriveAgent.get_optimizers
# hardcodes WarmupCosLR(epochs=200), so the cosine only completes at exactly 200.
# A shorter run stops near peak lr; a longer one makes the cosine turn back up.
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

# ---- data roots ----
# NEW-VLM (LLM-only LoRA = vlm_simscale_lora_merged) agent hidden-state caches.
SIM_AGENT_CACHE_ROOT="${SIM_AGENT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-${SIM_AGENT_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"
# Pre-built *_quality symlink views (round0 AND round1). Round1's quality tree is NOT
# under SIM_AGENT_CACHE_ROOT; it lives here on CPFS (same layout as fullmix / scene-router).
SIM_QUALITY_CACHE_ROOT="${SIM_QUALITY_CACHE_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale/new_vlm_quality_views}"
# Bucket token lists (quality-filtered, representation-INDEPENDENT). CPFS copy.
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"
NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR:-/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain}"

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
  # Prefer local *_quality, then CPFS quality views (has round1), else full cache.
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

# Per-teacher workspace, isolated so teachers never share symlink caches.
# Distinct from the RL script's MIX_ROOT so IL and RL views never collide.
MIX_ROOT="${MIX_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale/il_goal_training_newvlm/${BUCKET_NAME}_fullmix}"
NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH:-${MIX_ROOT}/navtrain_bucket_cache}"
SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH:-${MIX_ROOT}/simscale_bucket_cache}"
MIX_INFO_DIR="${MIX_INFO_DIR:-${MIX_ROOT}/metadata}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_teacher_${BUCKET_NAME}_il_goal_${GOAL_MODE}_newvlm}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/python}"

mkdir -p "${NAV_BUCKET_CACHE_PATH}" "${SIM_BUCKET_CACHE_PATH}" "${MIX_INFO_DIR}"

export MIX_NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR}"
export MIX_NAV_CACHE_PATH="${NAV_CACHE_PATH}"
export MIX_NAV_BUCKET_CACHE_PATH="${NAV_BUCKET_CACHE_PATH}"
export MIX_SIM_BUCKET_CACHE_PATH="${SIM_BUCKET_CACHE_PATH}"
export MIX_INFO_DIR="${MIX_INFO_DIR}"
export MIX_BUCKET_FILE="${BUCKET_FILE}"

IFS=','; export MIX_SIM_CACHE_PATHS="${SIM_CACHE_PATHS[*]}"; unset IFS
IFS=','; export MIX_SIM_BUCKET_DIRS="${SIM_BUCKET_DIRS[*]}"; unset IFS

# Builds the per-bucket symlink views over the agent caches. Unlike the RL script
# there is no union metric cache to assemble: IL has no reward, so the metric
# caches are not read at all.
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
bucket_file = os.environ.get("MIX_BUCKET_FILE", "exclusive_general_or_no_tag_tokens.json")

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
    "stage": "il_goal",
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
save_json(summary, info_dir / "bucket_il_goal_sources_summary.json")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PYPREP

echo "======================================================================"
echo "[teacher-il] BUCKET_NAME=${BUCKET_NAME}  BUCKET_FILE=${BUCKET_FILE}"
echo "[teacher-il] GOAL_MODE=${GOAL_MODE}  (privileged goal point = GT 8th waypoint)"
echo "[teacher-il] MIXTURE: full-mix union of navtrain_bucket + simscale_bucket (no ratios)"
echo "[teacher-il] BASE_CKPT=${BASE_CKPT:-<random-init DiT>}"
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
  agent=recogdrive_goal_agent \
  agent.goal_mode="${GOAL_MODE}" \
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

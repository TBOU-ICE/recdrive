#!/usr/bin/env bash
# Full-mix (navtrain + SimScale round0/round1) GOAL-CONDITIONED DiT IL launcher for the
# NEW (SimScale-LoRA) VLM hidden-state caches.
#
# Scene-agnostic counterpart of the four run_recogdrive_bucket_expert_*_il_goal_newvlm.sh
# teachers: same goal-point conditioning, but trained on the full mixture instead of one
# bucket. Use it to compare the three injection paths (adaln / channel / cross) without a
# bucket confound, and to produce the goal-free control with GOAL_MODE=none.
#
# Mixture: NO sampling ratios. mixed_cache.fullmix=True shuffles the UNION of the navtrain
# cache and every SimScale round with uniform per-sample probability, so the sources appear
# in their natural proportion and no WeightedRandomSampler is built.
#
# The goal point is PRIVILEGED information: the ground-truth 8th waypoint, read from
# targets["trajectory"][:, -1, :]. It is available only during training and during oracle
# evaluation; the student distilled from these teachers never sees it.
#
# GOAL_MODE (see navsim/agents/recogdrive/goal_cond.py):
#   adaln   - added to the AdaLN conditioning vector, next to the ego status
#   channel - added into the residual stream next to the fused DiT input
#   cross   - appended as one extra cross-attention key/value token
#   none    - no goal, i.e. the plain IL control group
#
# Differences vs run_recogdrive_train_fullmix_simscale_2b.sh:
#   * Reads full caches from SRC_CACHE_ROOT / OUT_ROOT (new-VLM, LLM-only LoRA caches --
#     the same representation the four bucket-expert teachers and the eval scripts use).
#   * Reuses a '*_quality' view next to the source cache when one exists; otherwise builds
#     the symlink view on QUALITY_CACHE_ROOT (CPFS) to avoid Alluxio FUSE EIO when creating
#     many mkdir/symlink metadata ops.
#   * Allowlist stays at ALLOWLIST_ROOT (unchanged).
#   * Starts a FRESH training: BASE_CKPT and CKPT_PATH are both empty, so the DiT is
#     random-init and nothing resumes from the old-VLM run.
#   * Defaults VLM_PATH to vlm_simscale_lora_merged and uses a distinct EXPERIMENT_NAME.
#
# Usage:
#   GOAL_MODE=adaln bash scripts/training/run_recogdrive_train_fullmix_simscale_goal_newvlm_2b.sh
#   # Compare all three injection paths plus the control:
#   for M in adaln channel cross none; do
#     GOAL_MODE=$M EXPERIMENT_NAME=il_goal_$M \
#       bash scripts/training/run_recogdrive_train_fullmix_simscale_goal_newvlm_2b.sh
#   done
#   # Skip quality filter (train on full SimScale caches):
#   USE_QUALITY_CACHE=false bash scripts/training/run_recogdrive_train_fullmix_simscale_goal_newvlm_2b.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PATH="${CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export CACHE_READ_MAX_RETRIES="${CACHE_READ_MAX_RETRIES:-5}"
export CACHE_READ_RETRY_BASE_SEC="${CACHE_READ_RETRY_BASE_SEC:-0.5}"

# ---- cache locations (NEW VLM, LLM-only LoRA = vlm_simscale_lora_merged) ----
# Full hidden-state caches produced by VLM_PATH below. This is the SAME representation
# the four bucket-expert teachers train on, and the one the eval scripts load, so the
# DiT trained here can be compared against them directly. The '..._vit_...' caches are a
# DIFFERENT representation and pair with vlm_simscale_lora_vit_merged -- do not mix them.
OUT_ROOT="${OUT_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_hidden_state_nav_sim}"
SRC_CACHE_ROOT="${SRC_CACHE_ROOT:-${OUT_ROOT}}"
# Fallback location for quality symlink views, used only when SRC_CACHE_ROOT ships no
# '*_quality' tree of its own. Kept on CPFS (not Alluxio) to avoid FUSE EIO.
QUALITY_CACHE_ROOT="${QUALITY_CACHE_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale/new_vlm_quality_views}"
# Backward-compat: CACHE_ROOT overrides where quality views live when set.
CACHE_ROOT="${CACHE_ROOT:-${QUALITY_CACHE_ROOT}}"
ALLOWLIST_ROOT="${ALLOWLIST_ROOT:-/workspace/datasets/simscale/20260709}"
SIM_ROUNDS="${SIM_ROUNDS:-0,1}"
USE_QUALITY_CACHE="${USE_QUALITY_CACHE:-true}"

NAV_CACHE_PATH="${NAV_CACHE_PATH:-${SRC_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"

IFS=',' read -r -a SIM_ROUND_LIST <<< "${SIM_ROUNDS}"
SIM_CACHE_PATHS=()
SIM_CACHE_NAMES=()
# Rounds whose quality view still has to be built under CACHE_ROOT (comma separated).
PREP_ROUNDS=""
FORCE_PREP_QUALITY_CACHE="${FORCE_PREP_QUALITY_CACHE:-false}"
for round in "${SIM_ROUND_LIST[@]}"; do
  round="${round//[[:space:]]/}"
  [[ -z "${round}" ]] && continue
  dataset_name="synthetic_reaction_pdm_v1.0-${round}"
  if [[ "${USE_QUALITY_CACHE}" == "true" ]]; then
    src_quality="${SRC_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}_quality"
    dst_quality="${CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}_quality"
    if [[ "${FORCE_PREP_QUALITY_CACHE}" != "true" ]] \
       && [[ -d "${src_quality}" ]] && [[ -n "$(ls -A "${src_quality}" 2>/dev/null || true)" ]]; then
      # The quality view already ships next to the source cache (same layout the four
      # bucket-expert teachers read); use it directly and skip the symlink prep.
      sim_cache_path="${src_quality}"
    else
      sim_cache_path="${dst_quality}"
      if [[ "${FORCE_PREP_QUALITY_CACHE}" == "true" ]] \
         || [[ ! -d "${dst_quality}" ]] || [[ -z "$(ls -A "${dst_quality}" 2>/dev/null || true)" ]]; then
        PREP_ROUNDS="${PREP_ROUNDS}${PREP_ROUNDS:+,}${round}"
      fi
    fi
  else
    sim_cache_path="${SRC_CACHE_ROOT}/recogdrive_agent_cache_dir_${dataset_name}"
  fi
  SIM_CACHE_PATHS+=("${sim_cache_path}")
  SIM_CACHE_NAMES+=("simscale_r${round}")
done
if [[ ${#SIM_CACHE_PATHS[@]} -eq 0 ]]; then
  echo "[ERROR] SIM_ROUNDS resolved to no cache paths: ${SIM_ROUNDS}" >&2
  exit 1
fi

# Weight-only warm start (agent.initialize). Empty = random-init DiT, the intended
# default here: this run trains the scene-agnostic goal DiT from scratch.
BASE_CKPT="${BASE_CKPT:-}"
# New merged VLM (LLM-only LoRA on SimScale). MUST match what produced the caches above.
VLM_PATH="${VLM_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/vlm_simscale_lora_merged}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Goal-point conditioning: adaln | channel | cross | none
GOAL_MODE="${GOAL_MODE:-cross}"

MASTER_PORT="${MASTER_PORT:-23561}"
GPUS="${GPUS:-8}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_dit_il_fullmix_simscale_goal_${GOAL_MODE}_newvlm}"
# 200 is deliberate: ReCogDriveAgent.get_optimizers hardcodes WarmupCosLR(epochs=200),
# so the cosine only anneals to min_lr at exactly 200. A shorter run stops near peak
# lr; a longer one drives the cosine past pi and the lr climbs back up.
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/python}"
# FRESH training by default (do NOT resume the old-VLM DiT). Set CKPT_PATH to resume.
CKPT_PATH="${CKPT_PATH:-}"

# ---- build quality symlink caches (only for rounds that need one) ----
# src: SRC_CACHE_ROOT (full caches)
# dst: CACHE_ROOT / QUALITY_CACHE_ROOT (CPFS symlink views)
# allowlist: ALLOWLIST_ROOT
if [[ "${USE_QUALITY_CACHE}" == "true" ]]; then
  if [[ -n "${PREP_ROUNDS}" ]]; then
    export PREP_SRC_CACHE_ROOT="${SRC_CACHE_ROOT}"
    export PREP_QUALITY_CACHE_ROOT="${CACHE_ROOT}"
    export PREP_ALLOWLIST_ROOT="${ALLOWLIST_ROOT}"
    export PREP_SIM_ROUNDS="${PREP_ROUNDS}"
    "${PYTHON_BIN}" - <<'PYPREP'
import os
from pathlib import Path

src_root = Path(os.environ["PREP_SRC_CACHE_ROOT"]).resolve()
dst_root = Path(os.environ["PREP_QUALITY_CACHE_ROOT"])
allow_root = Path(os.environ["PREP_ALLOWLIST_ROOT"])
rounds = [r.strip() for r in os.environ["PREP_SIM_ROUNDS"].split(",") if r.strip()]

for round in rounds:
    dataset_name = f"synthetic_reaction_pdm_v1.0-{round}"
    src_cache = src_root / f"recogdrive_agent_cache_dir_{dataset_name}"
    dst_cache = dst_root / f"recogdrive_agent_cache_dir_{dataset_name}_quality"
    allowlist = allow_root / f"quality_filter_{dataset_name}" / "allowlist_log_token.tsv"
    if not src_cache.is_dir():
        raise RuntimeError(f"SimScale source cache not found: {src_cache}")
    if not allowlist.is_file():
        raise RuntimeError(f"Quality allowlist not found: {allowlist}")

    linked = 0
    missing_src = 0
    errors = 0
    dst_cache.mkdir(parents=True, exist_ok=True)
    with allowlist.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            log_name, token = line.split("\t", 1)
            src = (src_cache / log_name / token).resolve()
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
                # Absolute symlink so dst on CPFS can point at Alluxio src.
                dst.symlink_to(src, target_is_directory=True)
            except (FileExistsError, FileNotFoundError, OSError) as exc:
                if dst.exists() or dst.is_symlink():
                    linked += 1
                    continue
                errors += 1
                if errors <= 5:
                    print(f"[fullmix-goal] symlink failed ({errors}): {dst} -> {src}: {exc}")
                continue
            linked += 1

    if linked == 0:
        raise RuntimeError(f"No quality cache entries linked for round {round}: {dst_cache}")
    print(f"[fullmix-goal] round={round} linked_quality_cache={linked} "
          f"missing_src={missing_src} errors={errors} src={src_cache} dst={dst_cache} allowlist={allowlist}")
PYPREP
  else
    echo "[fullmix-goal] quality caches already available for all rounds; skip prep (FORCE_PREP_QUALITY_CACHE=true to rebuild)"
  fi
fi

MIXED_CACHE_PATHS="${NAV_CACHE_PATH}"
MIXED_CACHE_NAMES="navtrain"
for idx in "${!SIM_CACHE_PATHS[@]}"; do
  MIXED_CACHE_PATHS="${MIXED_CACHE_PATHS},${SIM_CACHE_PATHS[$idx]}"
  MIXED_CACHE_NAMES="${MIXED_CACHE_NAMES},${SIM_CACHE_NAMES[$idx]}"
done

echo "[fullmix-goal] GOAL_MODE=${GOAL_MODE}  (privileged goal point = GT 8th waypoint)"
echo "[fullmix-goal] VLM_PATH=${VLM_PATH}"
echo "[fullmix-goal] SRC_CACHE_ROOT=${SRC_CACHE_ROOT}"
echo "[fullmix-goal] QUALITY/CACHE_ROOT=${CACHE_ROOT}"
echo "[fullmix-goal] ALLOWLIST_ROOT=${ALLOWLIST_ROOT}"
echo "[fullmix-goal] NAV_CACHE_PATH=${NAV_CACHE_PATH}"
for idx in "${!SIM_CACHE_PATHS[@]}"; do
  echo "[fullmix-goal] SIM_CACHE_PATH[${SIM_CACHE_NAMES[$idx]}]=${SIM_CACHE_PATHS[$idx]}"
done
echo "[fullmix-goal] MIXTURE: full-mix union of navtrain + all SimScale rounds (no ratios)"
echo "[fullmix-goal] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[fullmix-goal] CKPT_PATH=${CKPT_PATH:-<fresh>}  BASE_CKPT=${BASE_CKPT:-<random-init DiT>}"
echo "[fullmix-goal] GPUS=${GPUS} NNODES=${NNODES} RANK=${RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT}"

# Fail fast if a required cache is missing/empty.
for p in "${NAV_CACHE_PATH}" "${SIM_CACHE_PATHS[@]}"; do
  if [[ ! -d "${p}" ]] || [[ -z "$(ls -A "${p}" 2>/dev/null || true)" ]]; then
    echo "[ERROR] cache missing or empty: ${p}" >&2
    exit 1
  fi
done
if [[ -n "${CKPT_PATH}" && ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH does not exist: ${CKPT_PATH}" >&2
  exit 1
fi

HYDRA_ARGS=(
  agent=recogdrive_goal_agent
  agent.goal_mode="${GOAL_MODE}"
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

#!/usr/bin/env bash
# =============================================================================
# EPDMS GRPO scene-expert training (v1 plan), parametrized by BUCKET.
# Called by the four thin wrappers run_recogdrive_expert_epdms_{bucket}.sh.
#
# Intended launch = ONE expert per machine, single-node 8-GPU DDP:
#   # machine A
#   bash run_recogdrive_expert_epdms_general.sh
#   # machine B
#   bash run_recogdrive_expert_epdms_progress.sh
#   # machine C / D -> rule / safety
# Do NOT point WORLD_SIZE/RANK at a 4-machine job; the four experts are four
# independent jobs. Multi-node for a *single* expert is opt-in via NNODES.
#
# Representation: NEW-VLM-VIT (vlm_simscale_lora_vit_merged) + matching caches.
# Prerequisite: navtrain v2 metric cache built by
#   scripts/data/run_metric_caching_navtrain_v2.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"

export PATH="${CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Avoid dataloader fork deadlocks when many workers hit CPFS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

# Single-node by default. Prefer explicit NNODES/NODE_RANK — cluster injects of
# WORLD_SIZE/RANK (job-array size, etc.) previously made torchrun wait forever.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23481}"
GPUS="${GPUS:-8}"

# ---- expert identity ----
BUCKET_NAME="${BUCKET_NAME:-general_or_no_tag}"
SCENE_TERM="${SCENE_TERM:-none}"           # none | ep | ec | gate
SCENE_TERM_WEIGHT="${SCENE_TERM_WEIGHT:-0.3}"

# ---- sample-level source ratios ----
RATIO_NAV_FULL="${RATIO_NAV_FULL:-0.0}"
RATIO_NAV_BUCKET="${RATIO_NAV_BUCKET:-0.6}"
RATIO_SIM_BUCKET="${RATIO_SIM_BUCKET:-0.4}"

# ---- schedule (identical across experts on purpose) ----
LR="${LR:-2e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"             # SAMPLES per gpu-batch (a pair = 2)
NUM_WORKERS="${NUM_WORKERS:-8}"           # per-rank; 8gpu * 16 ≈ 128 loader procs
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SAMPLE_TIME="${SAMPLE_TIME:-8}"
BC_COEFF="${BC_COEFF:-0.1}"

# ---- new-vlm-VIT representation (VLM must match caches AND init ckpt) ----
VLM_PATH="${VLM_PATH:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/vlm_simscale_lora_vit_merged}"
VIT_CACHE_ROOT="${VIT_CACHE_ROOT:-/workspace/datasets/simscale/20260709/new_vlm_vit_hidden_state_nav_sim}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-${VIT_CACHE_ROOT}/recogdrive_agent_cache_dir_train}"
INIT_CKPT="${INIT_CKPT:-/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_dit_il_fullmix_simscale_newvlm-vit/version_0/checkpoints/epoch=199-step=312200.ckpt}"
REF_CKPT="${REF_CKPT:-${INIT_CKPT}}"

# ---- reward-side metric caches ----
V2_METRIC_CACHE="${V2_METRIC_CACHE:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/metric_cache_train_v2}"
SIMSCALE_METRIC_ROOT="${SIMSCALE_METRIC_ROOT:-/workspace/datasets/simscale/20260709}"
SIMSCALE_BUCKET_ROOT="${SIMSCALE_BUCKET_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"
NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR:-/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain}"
PAIR_TABLE="${PAIR_TABLE:-${REPO_ROOT}/data/epdms/navtrain_adjacent_pairs.json}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/data/epdms/manifests}"
NAV_MANIFEST="${NAV_MANIFEST:-${MANIFEST_DIR}/nav_train_vit.json}"
SIM_MANIFEST_R0="${SIM_MANIFEST_R0:-${MANIFEST_DIR}/sim_round0_vit.json}"
SIM_MANIFEST_R1="${SIM_MANIFEST_R1:-${MANIFEST_DIR}/sim_round1_vit.json}"
SIM_ROUNDS="${SIM_ROUNDS:-0,1}"

BUCKET_TOKENS="${NAVTRAIN_OUTPUT_DIR}/exclusive_${BUCKET_NAME}_tokens.json"

# resolve SimScale per-round agent caches / token lists / v1 metric caches
IFS=',' read -r -a ROUND_LIST <<< "${SIM_ROUNDS}"
SIM_CACHE_PATHS=(); SIM_TOKEN_LISTS=(); SIM_METRIC_DIRS=()
for r in "${ROUND_LIST[@]}"; do
  r="${r//[[:space:]]/}"; [[ -z "${r}" ]] && continue
  ds="synthetic_reaction_pdm_v1.0-${r}"
  SIM_CACHE_PATHS+=("${VIT_CACHE_ROOT}/recogdrive_agent_cache_dir_${ds}")
  SIM_TOKEN_LISTS+=("${SIMSCALE_BUCKET_ROOT}/scene_buckets_${ds}_quality/exclusive_${BUCKET_NAME}_tokens.json")
  SIM_METRIC_DIRS+=("${SIMSCALE_METRIC_ROOT}/metric_cache_${ds}")
done

# per-expert workspace: union CSV of the (v1) SimScale metric caches for this bucket
MIX_ROOT="${MIX_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale/epdms_rl_workspace/${BUCKET_NAME}}"
UNION_METRIC_DIR="${MIX_ROOT}/sim_metric_union"
mkdir -p "${UNION_METRIC_DIR}/metadata"

IFS=','; export PREP_SIM_METRIC_DIRS="${SIM_METRIC_DIRS[*]}"; unset IFS
IFS=','; export PREP_SIM_TOKEN_LISTS="${SIM_TOKEN_LISTS[*]}"; unset IFS
export PREP_UNION_METRIC_DIR="${UNION_METRIC_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun}"

"${PYTHON_BIN}" - <<'PYPREP'
import csv, json, os
from pathlib import Path

def norm(t):
    t = str(t).strip().lower()
    base, sep, sfx = t.rpartition("-")
    if sep and sfx.isdigit() and len(sfx) == 3 and base:
        return t
    return t.replace("-", "")

metric_dirs = [Path(p) for p in os.environ["PREP_SIM_METRIC_DIRS"].split(",") if p.strip()]
token_lists = [Path(p) for p in os.environ["PREP_SIM_TOKEN_LISTS"].split(",") if p.strip()]
out_dir = Path(os.environ["PREP_UNION_METRIC_DIR"])

allowed = set()
for tl in token_lists:
    if not tl.is_file():
        raise RuntimeError(f"missing SimScale bucket token list: {tl}")
    allowed |= {norm(t) for t in json.load(tl.open())}

# NOTE: no per-file existence check here -- the metadata CSVs are written by the
# metric caching jobs themselves, and stat-ing ~50k files on CPFS takes ~10 min.
rows, seen = [], set()
for md in metric_dirs:
    meta = md / "metadata"
    if not meta.is_dir():
        raise RuntimeError(f"missing metadata dir: {meta}")
    for csv_path in sorted(meta.glob("*.csv")):
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                p = row.get("file_name") or next(iter(row.values()))
                if not p:
                    continue
                token = norm(Path(p).parts[-2])
                if token in allowed and token not in seen:
                    rows.append(p); seen.add(token)

out_csv = out_dir / "metadata" / "sim_union_metric_cache_metadata_node_0.csv"
with out_csv.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["file_name"]); [w.writerow([p]) for p in rows]
print(f"[prep] SimScale v1 metric union: {len(rows)} tokens -> {out_csv}")
PYPREP

EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_expert_epdms_rl_${BUCKET_NAME}}"
SIM_CACHES_HYDRA="[$(IFS=,; echo "${SIM_CACHE_PATHS[*]}")]"
SIM_TOKENS_HYDRA="[$(IFS=,; echo "${SIM_TOKEN_LISTS[*]}")]"

echo "======================================================================"
echo "[epdms-rl] BUCKET=${BUCKET_NAME} SCENE_TERM=${SCENE_TERM}(w=${SCENE_TERM_WEIGHT})"
echo "[epdms-rl] launch: NNODES=${NNODES} NODE_RANK=${NODE_RANK} GPUS=${GPUS} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "[epdms-rl] ratios nav_full=${RATIO_NAV_FULL} nav_bucket=${RATIO_NAV_BUCKET} sim=${RATIO_SIM_BUCKET}"
echo "[epdms-rl] LR=${LR} epochs=${MAX_EPOCHS} (warmup ${WARMUP_EPOCHS}) steps/epoch=${STEPS_PER_EPOCH} batch=${BATCH_SIZE} workers=${NUM_WORKERS} prefetch=${PREFETCH_FACTOR} G=${SAMPLE_TIME}"
echo "[epdms-rl] INIT=${INIT_CKPT}"
echo "[epdms-rl] V2_METRIC_CACHE=${V2_METRIC_CACHE}"
echo "[epdms-rl] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "======================================================================"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_expert_epdms_rl.py" \
  agent=recogdrive_agent_epdms_rl \
  "agent.checkpoint_path='${INIT_CKPT}'" \
  "agent.reference_policy_checkpoint='${REF_CKPT}'" \
  "agent.vlm_path='${VLM_PATH}'" \
  agent.lr="${LR}" \
  "agent.metric_cache_path='${UNION_METRIC_DIR}'" \
  "agent.epdms_metric_cache_v2_path='${V2_METRIC_CACHE}'" \
  agent.scene_term="${SCENE_TERM}" \
  agent.scene_term_weight="${SCENE_TERM_WEIGHT}" \
  agent.rl_sample_time="${SAMPLE_TIME}" \
  agent.rl_bc_coeff="${BC_COEFF}" \
  agent.rl_max_epochs="${MAX_EPOCHS}" \
  agent.rl_warmup_epochs="${WARMUP_EPOCHS}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.check_val_every_n_epoch=3 \
  trainer.params.num_sanity_val_steps=0 \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  dataloader.params.pin_memory=true \
  dataloader.params.prefetch_factor="${PREFETCH_FACTOR}" \
  "+dataloader.params.persistent_workers=true" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  cache_path="${NAV_CACHE_PATH}" \
  use_cache_without_dataset=true \
  force_cache_computation=False \
  "+epdms.pair_table='${PAIR_TABLE}'" \
  "+epdms.bucket_tokens='${BUCKET_TOKENS}'" \
  "+epdms.sim_cache_paths=${SIM_CACHES_HYDRA}" \
  "+epdms.sim_token_lists=${SIM_TOKENS_HYDRA}" \
  "+epdms.nav_manifest='${NAV_MANIFEST}'" \
  "+epdms.sim_manifests=['${SIM_MANIFEST_R0}','${SIM_MANIFEST_R1}']" \
  "+epdms.ratio_nav_full=${RATIO_NAV_FULL}" \
  "+epdms.ratio_nav_bucket=${RATIO_NAV_BUCKET}" \
  "+epdms.ratio_sim_bucket=${RATIO_SIM_BUCKET}" \
  "+epdms.steps_per_epoch=${STEPS_PER_EPOCH}" \
  hydra/job_logging=stdout \
  hydra.output_subdir=null \
  "$@"

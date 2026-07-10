#!/usr/bin/env bash
set -euo pipefail

# Mixed navtrain + SimScale bucket teacher RL launcher.
# No Python training-code changes are required. Before launching training it builds:
#   1) a union feature cache with symlinks to navtrain/simscale token caches;
#   2) a union metric-cache metadata CSV pointing to navtrain/simscale metric caches;
#   3) generated bucket token JSONs where the target bucket is nav bucket + sampled SimScale.

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME is required: progress_curbside_stopgo|rule_intersection|safety_dynamics_interaction|general_or_no_tag}"
ROUND="${ROUND:-0}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}"

case "${BUCKET_NAME}" in
  progress_curbside_stopgo)
    DEFAULT_BUCKET_JSON="exclusive_progress_curbside_stopgo_tokens.json"
    DEFAULT_INIT_CKPT="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_bucket_progress_direct_rl/2026.07.01.22.28.37/lightning_logs/version_0/checkpoints/epoch=9-step=9350.ckpt"
    DEFAULT_LR="5e-5"
    DEFAULT_FULL_RATIO="0.25"
    DEFAULT_SIM_RATIO="0.20"
    ;;
  rule_intersection)
    DEFAULT_BUCKET_JSON="exclusive_rule_intersection_tokens.json"
    DEFAULT_INIT_CKPT="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_bucket_rule_direct_rl/2026.07.01.22.17.22/lightning_logs/version_0/checkpoints/epoch=9-step=10670.ckpt"
    DEFAULT_LR="5e-5"
    DEFAULT_FULL_RATIO="0.20"
    DEFAULT_SIM_RATIO="0.20"
    ;;
  safety_dynamics_interaction)
    DEFAULT_BUCKET_JSON="exclusive_safety_dynamics_interaction_tokens.json"
    DEFAULT_INIT_CKPT="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/training_recogdrive_bucket_safety_direct_dit_il_epoch44_rl/2026.07.02.05.57.03/lightning_logs/version_0/checkpoints/epoch=9-step=11580.ckpt"
    DEFAULT_LR="5e-5"
    DEFAULT_FULL_RATIO="0.15"
    DEFAULT_SIM_RATIO="0.20"
    ;;
  general_or_no_tag)
    DEFAULT_BUCKET_JSON="exclusive_general_or_no_tag_tokens.json"
    DEFAULT_INIT_CKPT="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"
    DEFAULT_LR="5e-5"
    DEFAULT_FULL_RATIO="0.50"
    DEFAULT_SIM_RATIO="0.25"
    ;;
  *)
    echo "[ERROR] Unknown BUCKET_NAME=${BUCKET_NAME}" >&2
    exit 1
    ;;
esac

export PATH="${CONDA_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin}:$PATH"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/recdrive-scene}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recogdrive/download}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is empty}"
MASTER_PORT="${MASTER_PORT:-23456}"
GPUS="${GPUS:-8}"

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/datasets/simscale/20260709}"
NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR:-/workspace/volumes/ad-e2e-al-sh01/nby/data/navtrain_scene/output/navtrain}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/exp/recogdrive_agent_cache_dir_train}"
SIM_CACHE_PATH="${SIM_CACHE_PATH:-${SIMSCALE_ROOT}/recogdrive_agent_cache_dir_${DATASET_NAME}}"
NAV_METRIC_CACHE_PATH="${NAV_METRIC_CACHE_PATH:-/mnt/volumes/ad-e2e-al-sh01/jiaoqf/recdrive/exp/metric_cache_train}"
SIM_METRIC_CACHE_PATH="${SIM_METRIC_CACHE_PATH:-${SIMSCALE_ROOT}/metric_cache_${DATASET_NAME}}"
MIX_ROOT="${MIX_ROOT:-${SIMSCALE_ROOT}/mixed_training/${BUCKET_NAME}}"
MIX_CACHE_PATH="${MIX_CACHE_PATH:-${MIX_ROOT}/union_recogdrive_agent_cache}"
MIX_METRIC_CACHE_PATH="${MIX_METRIC_CACHE_PATH:-${MIX_ROOT}/union_metric_cache}"
MIX_BUCKET_DIR="${MIX_BUCKET_DIR:-${MIX_ROOT}/bucket_tokens}"

INIT_CKPT="${INIT_CKPT:-${DEFAULT_INIT_CKPT}}"
REF_CKPT="${REF_CKPT:-${INIT_CKPT}}"
LR="${LR:-${DEFAULT_LR}}"
FULL_RATIO="${FULL_RATIO:-${DEFAULT_FULL_RATIO}}"
SIM_RATIO="${SIM_RATIO:-${DEFAULT_SIM_RATIO}}"
BUCKET_RATIO="${BUCKET_RATIO:-$(python - <<'PYRATIO'
import os
full = float(os.environ['FULL_RATIO'])
print(max(0.0, 1.0 - full))
PYRATIO
)}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
VLM_PATH="${VLM_PATH:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/ReCogDrive-VLM-2B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_bucket_${BUCKET_NAME}_mixed_simscale_rl}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/volumes/ad-e2e-al-sh01/nby/recdrive/conda_envs/recdrive/bin/python}"

mkdir -p "${MIX_ROOT}" "${MIX_CACHE_PATH}" "${MIX_METRIC_CACHE_PATH}/metadata" "${MIX_BUCKET_DIR}"

export MIX_BUCKET_NAME="${BUCKET_NAME}"
export MIX_NAVTRAIN_OUTPUT_DIR="${NAVTRAIN_OUTPUT_DIR}"
export MIX_NAV_CACHE_PATH="${NAV_CACHE_PATH}"
export MIX_SIM_CACHE_PATH="${SIM_CACHE_PATH}"
export MIX_NAV_METRIC_CACHE_PATH="${NAV_METRIC_CACHE_PATH}"
export MIX_SIM_METRIC_CACHE_PATH="${SIM_METRIC_CACHE_PATH}"
export MIX_CACHE_PATH="${MIX_CACHE_PATH}"
export MIX_METRIC_CACHE_PATH="${MIX_METRIC_CACHE_PATH}"
export MIX_BUCKET_DIR="${MIX_BUCKET_DIR}"
export MIX_FULL_RATIO="${FULL_RATIO}"
export MIX_SIM_RATIO="${SIM_RATIO}"
export MIX_SEED="${MIX_SEED:-0}"

"${PYTHON_BIN}" - <<'PYMIX'
import csv
import json
import os
import random
from pathlib import Path

bucket_name = os.environ['MIX_BUCKET_NAME']
navtrain_output_dir = Path(os.environ['MIX_NAVTRAIN_OUTPUT_DIR'])
nav_cache = Path(os.environ['MIX_NAV_CACHE_PATH'])
sim_cache = Path(os.environ['MIX_SIM_CACHE_PATH'])
nav_metric = Path(os.environ['MIX_NAV_METRIC_CACHE_PATH'])
sim_metric = Path(os.environ['MIX_SIM_METRIC_CACHE_PATH'])
mix_cache = Path(os.environ['MIX_CACHE_PATH'])
mix_metric = Path(os.environ['MIX_METRIC_CACHE_PATH'])
mix_bucket_dir = Path(os.environ['MIX_BUCKET_DIR'])
full_ratio = float(os.environ['MIX_FULL_RATIO'])
sim_ratio = float(os.environ['MIX_SIM_RATIO'])
seed = int(os.environ['MIX_SEED'])

bucket_files = {
    'safety_dynamics_interaction': 'exclusive_safety_dynamics_interaction_tokens.json',
    'rule_intersection': 'exclusive_rule_intersection_tokens.json',
    'progress_curbside_stopgo': 'exclusive_progress_curbside_stopgo_tokens.json',
    'general_or_no_tag': 'exclusive_general_or_no_tag_tokens.json',
}

def load_json(path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_token(token):
    return str(token).strip().lower().replace('-', '')

nav_token_to_meta = load_json(navtrain_output_dir / 'navtrain_token_to_buckets.json')
nav_token_to_log = {}
for token, meta in nav_token_to_meta.items():
    norm = normalize_token(token)
    log_name = meta.get('log_name') if isinstance(meta, dict) else None
    if norm and log_name:
        nav_token_to_log[norm] = log_name

nav_bucket_tokens = {}
for name, filename in bucket_files.items():
    tokens = [normalize_token(t) for t in load_json(navtrain_output_dir / filename)]
    nav_bucket_tokens[name] = [t for t in tokens if t in nav_token_to_log]

target_nav_tokens = nav_bucket_tokens[bucket_name]
if not target_nav_tokens:
    raise RuntimeError(f'No navtrain target tokens for {bucket_name}')

sim_token_to_log = {}
if sim_cache.is_dir():
    for log_dir in sim_cache.iterdir():
        if not log_dir.is_dir():
            continue
        for token_dir in log_dir.iterdir():
            if not token_dir.is_dir():
                continue
            token = normalize_token(token_dir.name)
            if (token_dir / 'internvl_feature.gz').is_file() and (token_dir / 'trajectory_target.gz').is_file():
                sim_token_to_log[token] = log_dir.name

def read_metric_tokens(metric_root):
    metadata_dir = Path(metric_root) / 'metadata'
    tokens = set()
    paths = []
    if not metadata_dir.is_dir():
        return tokens, paths
    for csv_path in metadata_dir.glob('*.csv'):
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get('file_name') or next(iter(row.values()))
                if not path:
                    continue
                paths.append(path)
                parts = Path(path).parts
                if len(parts) >= 2:
                    tokens.add(normalize_token(parts[-2]))
    return tokens, paths

_, nav_metric_paths = read_metric_tokens(nav_metric)
sim_metric_tokens, sim_metric_paths = read_metric_tokens(sim_metric)
if sim_metric_tokens:
    sim_token_to_log = {t: l for t, l in sim_token_to_log.items() if t in sim_metric_tokens}

if not sim_token_to_log:
    raise RuntimeError(f'No usable SimScale tokens found. Check SIM_CACHE_PATH={sim_cache} and SIM_METRIC_CACHE_PATH={sim_metric}')

nav_target_count = len(target_nav_tokens)
nav_bucket_total_ratio = 1.0 - full_ratio - sim_ratio
if nav_bucket_total_ratio <= 0:
    raise RuntimeError(f'Invalid ratios: FULL_RATIO={full_ratio}, SIM_RATIO={sim_ratio}; need full+sim < 1')
requested_sim_count = int(round(nav_target_count * sim_ratio / nav_bucket_total_ratio))
requested_sim_count = max(1, requested_sim_count)
sim_tokens_all = sorted(sim_token_to_log)
rng = random.Random(seed)
sim_tokens = sorted(rng.sample(sim_tokens_all, requested_sim_count)) if requested_sim_count < len(sim_tokens_all) else sim_tokens_all

combined_token_to_log = dict(nav_token_to_log)
combined_token_to_log.update({t: sim_token_to_log[t] for t in sim_tokens})
save_json({t: {'log_name': log} for t, log in combined_token_to_log.items()}, mix_bucket_dir / 'mixed_token_to_log.json')

for name, tokens in nav_bucket_tokens.items():
    out_tokens = list(tokens)
    if name == bucket_name:
        out_tokens = sorted(set(out_tokens) | set(sim_tokens))
    save_json(out_tokens, mix_bucket_dir / bucket_files[name])

def link_token(root, log_name, token):
    src = root / log_name / token
    if not src.is_dir():
        return False
    dst_log = mix_cache / log_name
    dst_log.mkdir(parents=True, exist_ok=True)
    dst = dst_log / token
    if dst.exists() or dst.is_symlink():
        return True
    dst.symlink_to(src, target_is_directory=True)
    return True

linked_nav = 0
for tokens in nav_bucket_tokens.values():
    for token in tokens:
        log_name = nav_token_to_log.get(token)
        if log_name and link_token(nav_cache, log_name, token):
            linked_nav += 1

linked_sim = 0
for token in sim_tokens:
    log_name = sim_token_to_log[token]
    if link_token(sim_cache, log_name, token):
        linked_sim += 1

metric_csv = mix_metric / 'metadata' / 'mixed_metric_cache_metadata_node_0.csv'
metric_csv.parent.mkdir(parents=True, exist_ok=True)
with metric_csv.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name'])
    for path in nav_metric_paths:
        writer.writerow([path])
    selected_sim_set = set(sim_tokens)
    for path in sim_metric_paths:
        token = normalize_token(Path(path).parts[-2])
        if token in selected_sim_set:
            writer.writerow([path])

summary = {
    'bucket_name': bucket_name,
    'full_ratio': full_ratio,
    'bucket_ratio_passed_to_existing_trainer': 1.0 - full_ratio,
    'requested_total_sim_ratio': sim_ratio,
    'nav_target_bucket_tokens': len(target_nav_tokens),
    'selected_sim_tokens': len(sim_tokens),
    'available_sim_tokens': len(sim_tokens_all),
    'linked_nav_token_dirs': linked_nav,
    'linked_sim_token_dirs': linked_sim,
    'mix_cache_path': str(mix_cache),
    'mix_metric_cache_path': str(mix_metric),
    'mix_bucket_dir': str(mix_bucket_dir),
}
save_json(summary, mix_bucket_dir / 'summary.json')
print(json.dumps(summary, indent=2, ensure_ascii=False))
PYMIX

BUCKET_JSON="${MIX_BUCKET_DIR}/${DEFAULT_BUCKET_JSON}"
TOKEN_TO_LOG_JSON="${MIX_BUCKET_DIR}/mixed_token_to_log.json"

echo "BUCKET_NAME: ${BUCKET_NAME}"
echo "INIT_CKPT: ${INIT_CKPT}"
echo "REF_CKPT: ${REF_CKPT}"
echo "CACHE_PATH: ${MIX_CACHE_PATH}"
echo "METRIC_CACHE_PATH: ${MIX_METRIC_CACHE_PATH}"
echo "BUCKET_JSON: ${BUCKET_JSON}"
echo "TOKEN_TO_LOG_JSON: ${TOKEN_TO_LOG_JSON}"
echo "FULL_RATIO: ${FULL_RATIO}"
echo "BUCKET_RATIO: ${BUCKET_RATIO}"
echo "SIM_RATIO target used for sampling: ${SIM_RATIO}"

"${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_bucket_rl.py" \
  agent=recogdrive_agent \
  "agent.checkpoint_path=\"${INIT_CKPT}\"" \
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
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  cache_path="${MIX_CACHE_PATH}" \
  bucket.name="${BUCKET_NAME}" \
  bucket.tokens_json="${BUCKET_JSON}" \
  bucket.token_to_log_json="${TOKEN_TO_LOG_JSON}" \
  bucket.navtrain_output_dir="${MIX_BUCKET_DIR}" \
  bucket.full_ratio="${FULL_RATIO}" \
  bucket.bucket_ratio="${BUCKET_RATIO}" \
  hydra/job_logging=stdout \
  hydra.output_subdir=null

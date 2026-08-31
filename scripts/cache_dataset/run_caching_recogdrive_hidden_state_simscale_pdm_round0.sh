#!/usr/bin/env bash
set -euo pipefail

# Cache ReCogDrive VLM hidden states for SimScale round0 + round1 + navtrain in one run,
# all with the SAME VLM (pass VLM_PATH). This keeps the fullmix DiT training consistent
# (navtrain and simscale caches share one representation).
#
# IMPORTANT #1 - VLM: pass the MERGED new VLM, otherwise you re-cache with the old base:
#   VLM_PATH=/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged \
#     GPUS=8 bash scripts/cache_dataset/run_caching_recogdrive_hidden_state_simscale_pdm_round0.sh
#
# IMPORTANT #2 - OUTPUT: all three caches are written UNDER a single new root OUT_ROOT
# (default /workspace/datasets/simscale/20260709/data/new_vlm_vit_hidden_state_nav_sim), so
# the old-VLM caches at their canonical locations are NOT touched. The DiT training must then
# point its cache paths at OUT_ROOT.
#   OUT_ROOT/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-0   (simscale r0)
#   OUT_ROOT/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-1   (simscale r1)
#   OUT_ROOT/recogdrive_agent_cache_dir_train                           (navtrain)
#
# Knobs:
#   VLM_PATH         VLM to encode with (REQUIRED to be the new merged VLM for this exp)
#   OUT_ROOT         single parent dir for all three caches (default new path above)
#   ROUNDS           simscale rounds, space-separated (default "0 1")
#   CACHE_NAVTRAIN   1 (default) to also cache navtrain, 0 to skip
#   GPUS / CUDA_VISIBLE_DEVICES   GPUs to use
#   CACHE_ROOT       simscale cache parent (default OUT_ROOT)
#   NAV_CACHE_PATH   navtrain cache dir (default OUT_ROOT/recogdrive_agent_cache_dir_train)
#   NAVTRAIN_DATA_ROOT  navtrain OpenScene data root (default download dir)

ROUNDS="${ROUNDS:-${ROUND:-0 1}}"
CACHE_NAVTRAIN="${CACHE_NAVTRAIN:-1}"

# All three caches (simscale round0/round1 + navtrain) are written UNDER this single
# root so nothing overwrites the old-VLM caches at their canonical locations.
OUT_ROOT="${OUT_ROOT:-/workspace/datasets/simscale/20260709/data/new_vlm_vit_hidden_state_nav_sim}"

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/datasets/simscale/20260709}"
CACHE_ROOT="${CACHE_ROOT:-${OUT_ROOT}}"
# Hydra/experiment output must NOT be written under SIMSCALE_ROOT: that path is an
# Alluxio FUSE mount that intermittently throws OSError [Errno 5] (EIO) on the small
# metadata writes Hydra does (overrides.yaml/config.yaml). Keep exp output on CPFS.
SIMSCALE_EXP_ROOT="${SIMSCALE_EXP_ROOT:-${OUT_ROOT}/exp}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# navtrain locations (data root differs from simscale!)
NAVTRAIN_SPLIT="${NAVTRAIN_SPLIT:-navtrain}"
NAVTRAIN_DATA_ROOT="${NAVTRAIN_DATA_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download}"
NAVTRAIN_EXP_ROOT="${NAVTRAIN_EXP_ROOT:-/workspace/models/recdrive/v1.0.0}"
NAV_CACHE_PATH="${NAV_CACHE_PATH:-${OUT_ROOT}/recogdrive_agent_cache_dir_train}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
CONDA_PYTHON_ROOT="${CONDA_PYTHON_ROOT:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_PYTHON_ROOT}/bin/torchrun}"
VLM_PATH="${VLM_PATH:-/workspace/volumes/ad-e2e-bd-su01/nby/recdrive/vlm_simscale_lora_vit_merged}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-63669}"

detect_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
        local count=0
        local item
        IFS=',' read -ra visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
        for item in "${visible_devices[@]}"; do
            [[ -n "${item// /}" ]] && count=$((count + 1))
        done
        echo "${count}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
        return
    fi
    echo 0
}

DETECTED_GPUS="$(detect_gpus)"
GPUS_PER_NODE="${GPUS:-${DETECTED_GPUS}}"
if (( GPUS_PER_NODE < 1 )); then
    echo "[ERROR] No GPU detected. This hidden-state cache needs GPU (agent.cache_hidden_state=True)." >&2
    echo "        Set CUDA_VISIBLE_DEVICES or pass GPUS=<num_visible_gpus>." >&2
    exit 1
fi

export MASTER_ADDR MASTER_PORT
mkdir -p "${LOG_DIR}"

# ---- build the job list: simscale rounds (+ navtrain) ----
JOB_NAME=(); JOB_SPLIT=(); JOB_DATAROOT=(); JOB_EXPROOT=(); JOB_CACHE=()
for round in ${ROUNDS}; do
  round="${round//[[:space:]]/}"
  [[ -z "${round}" ]] && continue
  ds="synthetic_reaction_pdm_v1.0-${round}"
  JOB_NAME+=("simscale_r${round}")
  JOB_SPLIT+=("simscale_pdm_round${round}")
  JOB_DATAROOT+=("${SIMSCALE_ROOT}")
  JOB_EXPROOT+=("${SIMSCALE_EXP_ROOT}")
  JOB_CACHE+=("${CACHE_ROOT}/recogdrive_agent_cache_dir_${ds}")
done
if [[ "${CACHE_NAVTRAIN}" == "1" ]]; then
  JOB_NAME+=("navtrain")
  JOB_SPLIT+=("${NAVTRAIN_SPLIT}")
  JOB_DATAROOT+=("${NAVTRAIN_DATA_ROOT}")
  JOB_EXPROOT+=("${NAVTRAIN_EXP_ROOT}")
  JOB_CACHE+=("${NAV_CACHE_PATH}")
fi

echo "=================================================="
echo " ReCogDrive hidden-state caching (multi-target)"
echo "   VLM_PATH=${VLM_PATH}"
echo "   OUT_ROOT=${OUT_ROOT}"
echo "   jobs: ${JOB_NAME[*]}"
echo "   GPUS_PER_NODE=${GPUS_PER_NODE} (detected=${DETECTED_GPUS})"
echo "=================================================="
if [[ "${VLM_PATH}" == *"ReCogDrive-VLM-2B" ]]; then
  echo "[WARN] VLM_PATH looks like the OLD base VLM. For the SimScale-LoRA experiment pass"
  echo "       VLM_PATH=/workspace/models/recdrive/v1.0.0/vlm_simscale_lora_merged"
fi
for i in "${!JOB_NAME[@]}"; do
  if [[ -d "${JOB_CACHE[$i]}" ]] && [[ -n "$(ls -A "${JOB_CACHE[$i]}" 2>/dev/null || true)" ]]; then
    echo "[WARN] ${JOB_NAME[$i]} cache dir NOT empty -> will be OVERWRITTEN (force_cache_computation=true):"
    echo "       ${JOB_CACHE[$i]}"
  fi
done

# ---- run each job ----
idx=0
for i in "${!JOB_NAME[@]}"; do
  name="${JOB_NAME[$i]}"
  split="${JOB_SPLIT[$i]}"
  dataroot="${JOB_DATAROOT[$i]}"
  exproot="${JOB_EXPROOT[$i]}"
  cache="${JOB_CACHE[$i]}"
  port=$(( MASTER_PORT + idx ))
  idx=$(( idx + 1 ))
  logf="${LOG_DIR}/caching_recogdrive_hidden_state_${name}_rank${NODE_RANK}.txt"

  export OPENSCENE_DATA_ROOT="${dataroot}"
  export NAVSIM_EXP_ROOT="${exproot}"
  mkdir -p "${cache}" "${exproot}"

  echo ""
  echo "----- caching ${name}: split=${split} -----"
  echo "   OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
  echo "   CACHE_PATH=${cache}"
  echo "   MASTER_PORT=${port}"
  echo "   LOG_FILE=${logf}"

  "${TORCHRUN_BIN}" \
      --nnodes="${NNODES}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${port}" \
      --nproc_per_node="${GPUS_PER_NODE}" \
      "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py" \
      agent=recogdrive_agent \
      experiment_name="recogdrive_agent_cache_${name}" \
      agent.cam_type='single' \
      agent.cache_hidden_state=True \
      agent.cache_mode=True \
      train_test_split="${split}" \
      agent.vlm_path="${VLM_PATH}" \
      cache_path="${cache}" \
      worker=sequential \
      2>&1 | tee "${logf}"

  echo "----- done ${name} -> ${cache} -----"
done

echo ""
echo "All caching jobs done: ${JOB_NAME[*]}"
echo "All caches under OUT_ROOT=${OUT_ROOT}"
echo "  simscale: ${CACHE_ROOT}/recogdrive_agent_cache_dir_synthetic_reaction_pdm_v1.0-<round>"
[[ "${CACHE_NAVTRAIN}" == "1" ]] && echo "  navtrain: ${NAV_CACHE_PATH}"
echo "NEXT: point the DiT training's cache paths at OUT_ROOT (navtrain=NAV_CACHE_PATH, simscale under CACHE_ROOT)."

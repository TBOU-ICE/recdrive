#!/usr/bin/env bash
# Build the navtrain metric cache in **navsim_v2 (EPDMS) format**.
#
# Why: the RL reward switch to EPDMS needs the v2 scorer, which reads fields the
# old v1 cache does not have (map_parameters for LK/TLC, past_human_trajectory
# for HC, log_name/timepoint for adjacent-frame pairing). The existing
# /mnt/.../jiaoqf/recdrive/exp/metric_cache_train is v1-format and stays untouched.
#
# CPU-only (ray workers), no GPU. Expect several hours for ~100k frames.
#   bash scripts/data/run_metric_caching_navtrain_v2.sh
set -euo pipefail

export PATH="/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin:${PATH}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/exp"
export NAVSIM_DEVKIT_ROOT="/workspace/volumes/ad-e2e-bd-su01/nby/navsim_v2"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}"
export OPENSCENE_DATA_ROOT="/workspace/datasets/recdrive/20260513/nby/recdrive/download"

CACHE_PATH="${CACHE_PATH:-/workspace/datasets/recdrive/20260513/nby/recdrive/metric_cache_train_v2}"
THREADS="${THREADS:-64}"

mkdir -p "${CACHE_PATH}"

# IMPORTANT: run from the navsim_v2 repo. Ray workers resolve the `navsim`
# package from the current working directory first; launching from another
# repo makes them import that repo's (v1) caching module and crash.
cd "${NAVSIM_DEVKIT_ROOT}"

/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/python \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtrain \
  metric_cache_path="${CACHE_PATH}" \
  worker=ray_distributed_no_torch \
  worker.threads_per_node="${THREADS}"

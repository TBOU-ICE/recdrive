#!/usr/bin/env bash
# Build all four bucket IL direct-path indexes on this machine (CPU only).
# No cache data or symlinks are created. Training reads the JSON indexes and
# accesses the original cache paths directly.
#
#   bash scripts/training/prep_all_bucket_il_newvlm.sh
#   PREP_WORKERS=32 bash scripts/training/prep_all_bucket_il_newvlm.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/recdrive/bin/python}"
PREP_PY="${REPO_ROOT}/scripts/data/prep_bucket_il_direct_indexes.py"

echo "[prep-all] building full/bucket navtrain val indexes + 4 direct train indexes"
"${PYTHON_BIN}" "${PREP_PY}" --all-buckets --workers "${PREP_WORKERS:-16}"
echo "[prep-all] done. Train with SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_<bucket>_il_newvlm.sh"

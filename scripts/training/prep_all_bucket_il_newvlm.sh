#!/usr/bin/env bash
# Build all four no-goal bucket IL symlink views + reusable dataset indexes
# on this machine (CPU only). After this, training loads JSON indexes and
# does not walk cache trees.
#
#   bash scripts/training/prep_all_bucket_il_newvlm.sh
#   PREP_WORKERS=32 bash scripts/training/prep_all_bucket_il_newvlm.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/volumes/ad-e2e-bd-su01/nby/conda_envs/recdrive/bin/python}"
PREP_PY="${REPO_ROOT}/scripts/data/prep_bucket_il_newvlm.py"

echo "[prep-all] building navtrain val index + 4 bucket symlink views/indexes"
"${PYTHON_BIN}" "${PREP_PY}" --all-buckets --workers "${PREP_WORKERS:-16}"
echo "[prep-all] done. Train with SKIP_PREP=true GPUS=8 bash scripts/training/run_recogdrive_bucket_expert_<bucket>_il_newvlm.sh"

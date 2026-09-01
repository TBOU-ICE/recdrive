#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRIC=navhard exec bash "${SCRIPT_DIR}/run_privileged_opd_v2_eval.sh" "$@"

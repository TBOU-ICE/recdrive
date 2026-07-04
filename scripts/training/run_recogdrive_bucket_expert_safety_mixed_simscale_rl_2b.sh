#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUCKET_NAME="${BUCKET_NAME:-safety_dynamics_interaction}"
exec "${SCRIPT_DIR}/run_recogdrive_bucket_expert_mixed_simscale_rl_2b.sh" "$@"

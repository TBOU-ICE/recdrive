#!/usr/bin/env bash
# B: A + training-only Auxiliary Goal Internalization; predicted goal never enters planner.
set -euo pipefail
VARIANT=B GOAL_AUX_WEIGHT="${GOAL_AUX_WEIGHT:-0.2}" GOAL_PREF_WEIGHT=0 RESIDUAL_PRIVILEGE=false \
exec bash "$(dirname "$0")/run_opd_common.sh" "$@"

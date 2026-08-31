#!/usr/bin/env bash
# G: advanced combined variant = residual-privilege OPD + preference distillation + aux goal.
set -euo pipefail
VARIANT=G GOAL_AUX_WEIGHT="${GOAL_AUX_WEIGHT:-0.1}" GOAL_PREF_WEIGHT="${GOAL_PREF_WEIGHT:-0.2}" RESIDUAL_PRIVILEGE=true \
exec bash "$(dirname "$0")/run_opd_common.sh" "$@"

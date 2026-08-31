#!/usr/bin/env bash
# F: Residual-Privilege OPD. Target = ref + [teacher(goal ON)-teacher(goal OFF)].
set -euo pipefail
VARIANT=F GOAL_AUX_WEIGHT=0 GOAL_PREF_WEIGHT=0 RESIDUAL_PRIVILEGE=true \
exec bash "$(dirname "$0")/run_opd_common.sh" "$@"

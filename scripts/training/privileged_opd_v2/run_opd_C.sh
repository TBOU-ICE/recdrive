#!/usr/bin/env bash
# C: A + Goal Preference Distillation over an offline endpoint vocabulary.
set -euo pipefail
VARIANT=C GOAL_AUX_WEIGHT=0 GOAL_PREF_WEIGHT="${GOAL_PREF_WEIGHT:-0.2}" RESIDUAL_PRIVILEGE=false \
exec bash "$(dirname "$0")/run_opd_common.sh" "$@"

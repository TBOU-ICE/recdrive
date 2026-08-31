#!/usr/bin/env bash
# A: goal-free student + same-student-rollout routed reverse-KL; no goal head.
set -euo pipefail
VARIANT=A GOAL_AUX_WEIGHT=0 GOAL_PREF_WEIGHT=0 RESIDUAL_PRIVILEGE=false \
exec bash "$(dirname "$0")/run_opd_common.sh" "$@"

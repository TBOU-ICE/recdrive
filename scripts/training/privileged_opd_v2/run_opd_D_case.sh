#!/usr/bin/env bash
# D-case OPD helper. TEACHER_*_CKPT must point to four teachers trained with the same D_CASE.
set -euo pipefail
D_CASE="${D_CASE:-D6}"
case "$D_CASE" in
  D1) export GOAL_POINT_MODE=final;  export GOAL_INJECTION=adaln ;;
  D2) export GOAL_POINT_MODE=final;  export GOAL_INJECTION=cross ;;
  D3) export GOAL_POINT_MODE=final;  export GOAL_INJECTION=gated_cross ;;
  D4) export GOAL_POINT_MODE=multi3; export GOAL_INJECTION=adaln ;;
  D5) export GOAL_POINT_MODE=multi3; export GOAL_INJECTION=cross ;;
  D6) export GOAL_POINT_MODE=multi3; export GOAL_INJECTION=gated_cross ;;
  *) echo "Unknown D_CASE=$D_CASE (D1..D6)" >&2; exit 2 ;;
esac
export VARIANT="$D_CASE"
export GOAL_AUX_WEIGHT=0
export GOAL_PREF_WEIGHT=0
export RESIDUAL_PRIVILEGE=false
exec bash "$(dirname "$0")/run_opd_common.sh" "$@"

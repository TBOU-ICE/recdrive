#!/usr/bin/env bash
# D-case helper for ONE teacher. Usage:
#   D_CASE=D6 BUCKET_NAME=progress_curbside_stopgo BASE_RL_CKPT=$RL_PROGRESS_CKPT bash .../run_teacher_stage3_D_case.sh
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
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_priv_goal_v2_${BUCKET_NAME:-general_or_no_tag}_${D_CASE}}"
exec bash "$(dirname "$0")/run_teacher_stage3.sh" "$@"

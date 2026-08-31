#!/usr/bin/env bash
# D architecture ablation. Set GOAL_INJECTION and GOAL_POINT_MODE, and point the
# four TEACHER_*_CKPT vars at teachers trained with exactly the same settings.
set -euo pipefail
VARIANT="D_${GOAL_POINT_MODE:-final}_${GOAL_INJECTION:-gated_cross}" GOAL_AUX_WEIGHT=0 GOAL_PREF_WEIGHT=0 RESIDUAL_PRIVILEGE=false \
exec bash "$(dirname "$0")/run_opd_common.sh" "$@"

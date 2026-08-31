#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
need STAGE1_IL_CKPT
# Original recipe remains scene-specific RL: default 0.6 NAV bucket + 0.4 SimScale bucket.
INIT_CKPT="$STAGE1_IL_CKPT" exec bash "$NAVSIM_DEVKIT_ROOT/scripts/training/run_recogdrive_expert_epdms_rule.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${CONFIG:?Set CONFIG to your completed student_state_correction JSON file}"
STAGE="${1:?Usage: run_student_state_correction.sh collect|finetune|validate|opd|evaluate-student|plot [extra arguments]}"
shift
RUNNER="${REPO_ROOT}/navsim/planning/script/run_student_state_correction.py"
case "$STAGE" in
  collect|validate|evaluate-student)
    "${PYTHON_BIN:-python}" "$RUNNER" "$STAGE" --config "$CONFIG" "$@"
    ;;
  finetune|opd)
    "${TORCHRUN_BIN:-torchrun}" --standalone --nnodes=1 --nproc_per_node="${GPUS:-8}" \
      "$RUNNER" "$STAGE" --config "$CONFIG" "$@"
    ;;
  plot)
    "${PYTHON_BIN:-python}" "${REPO_ROOT}/scripts/visualization/plot_student_state_correction.py" \
      --config "$CONFIG" "$@"
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

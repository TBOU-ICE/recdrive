#!/usr/bin/env bash
# Launcher: wait 4 hours, then run NavSim 1.1 PDMS eval for SGDrive stage3 RL.
set -euo pipefail

SCRIPT="/workspace/recdrive/scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_1gpu_debug.sh"
LOG_DIR="/workspace/volumes/ad-e2e-al-sh01/nby/recdrive/exp/eval_recdrive"
WAIT_SECONDS=14400

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/scheduled_sgdrive_eval_$(date +%Y.%m.%d.%H.%M.%S).log"

echo "========================================"
echo "Scheduled SGDrive PDMS eval (NavSim 1.1)"
echo "Current time:  $(date)"
echo "Will start at: $(date -d "+${WAIT_SECONDS} seconds" 2>/dev/null || date -u -v+4H)"
echo "Script:        ${SCRIPT}"
echo "Log file:      ${LOG_FILE}"
echo "Wait seconds:  ${WAIT_SECONDS}"
echo "========================================"

sleep "${WAIT_SECONDS}"

echo "========================================"
echo "Starting eval at: $(date)"
echo "========================================"

bash "${SCRIPT}" 2>&1 | tee "${LOG_FILE}"

echo "========================================"
echo "Finished at: $(date)"
echo "Log: ${LOG_FILE}"
echo "========================================"

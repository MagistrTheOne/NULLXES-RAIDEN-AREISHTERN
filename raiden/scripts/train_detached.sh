#!/usr/bin/env bash
# Detached launch that survives SSH hangup (RunPod batch job).

set -euo pipefail

if [[ -f /workspace/raiden.env ]]; then
  # shellcheck disable=SC1091
  source /workspace/raiden.env
fi

export RAIDEN_ROOT="${RAIDEN_ROOT:-/workspace/raiden}"
LOG="${RAIDEN_LOGS:-/workspace/logs}/train_detached.log"
mkdir -p "$(dirname "$LOG")"

setsid bash -c "bash ${RAIDEN_ROOT}/scripts/train_raiden.sh" >"$LOG" 2>&1 </dev/null &
echo "LAUNCHED pid=$! log=$LOG"
echo "monitor: tail -n 50 $LOG"

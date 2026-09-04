#!/usr/bin/env bash
# Resume Stage I from the newest checkpoint-* on the volume.

set -euo pipefail

if [[ -f /workspace/raiden.env ]]; then
  # shellcheck disable=SC1091
  source /workspace/raiden.env
fi

export RAIDEN_ROOT="${RAIDEN_ROOT:-/workspace/raiden}"
export PYTHONPATH="${RAIDEN_ROOT}/src:${PYTHONPATH:-}"
export RAIDEN_RESUME="${RAIDEN_RESUME:-auto}"
cd "$RAIDEN_ROOT"

exec bash "${RAIDEN_ROOT}/scripts/train_raiden.sh"

#!/usr/bin/env bash
# Full RAIDEN eval suite. Compare adapter vs optional --base-only.

set -euo pipefail

if [[ -f /workspace/raiden.env ]]; then
  # shellcheck disable=SC1091
  source /workspace/raiden.env
fi

export RAIDEN_ROOT="${RAIDEN_ROOT:-/workspace/raiden}"
export PYTHONPATH="${RAIDEN_ROOT}/src:${PYTHONPATH:-}"
cd "$RAIDEN_ROOT"

OUT="${1:-${RAIDEN_LOGS:-/workspace/logs}/eval_report.json}"
ADAPTER="${2:-}"

ARGS=(--config configs/raiden_qlora.yaml --eval-config configs/eval.yaml --out "$OUT")
if [[ -n "$ADAPTER" ]]; then
  ARGS+=(--adapter "$ADAPTER")
fi

python3 scripts/eval_raiden.py "${ARGS[@]}"
echo "eval report: $OUT"

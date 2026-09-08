#!/usr/bin/env bash
# Stage I QLoRA SFT. Detach-friendly: logs on the volume.
# Does not start preference training.

set -euo pipefail

if [[ -f /workspace/raiden.env ]]; then
  # shellcheck disable=SC1091
  source /workspace/raiden.env
fi

export RAIDEN_ROOT="${RAIDEN_ROOT:-/workspace/raiden}"
export PYTHONPATH="${RAIDEN_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
cd "$RAIDEN_ROOT"

CONFIG="${RAIDEN_ROOT}/configs/raiden_qlora.yaml"
TRAIN_JSONL="${RAIDEN_DATASETS:-/workspace/datasets}/raiden/train.jsonl"
VAL_JSONL="${RAIDEN_DATASETS:-/workspace/datasets}/raiden/validation.jsonl"
LOG="${RAIDEN_LOGS:-/workspace/logs}/train_raiden.log"

mkdir -p "$(dirname "$LOG")" "$(dirname "$TRAIN_JSONL")"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "building dataset -> ${RAIDEN_DATASETS:-/workspace/datasets}/raiden"
  python3 scripts/prepare_dataset.py \
    --n-train "${RAIDEN_N_TRAIN:-24000}" \
    --n-val "${RAIDEN_N_VAL:-2000}" \
    --n-replay "${RAIDEN_N_REPLAY:-0}" \
    --out-dir "${RAIDEN_DATASETS:-/workspace/datasets}/raiden"
fi

python3 scripts/validate_dataset.py \
  --train "$TRAIN_JSONL" \
  --val "$VAL_JSONL" \
  --preference "${RAIDEN_DATASETS:-/workspace/datasets}/raiden/preference.jsonl" \
  --tol 0.10

python3 scripts/inspect_model.py --config "$CONFIG" --out "${RAIDEN_LOGS:-/workspace/logs}/inspect_pretrain.json"

python3 -m raiden.validate_runtime --config "$CONFIG"

echo "starting QLoRA SFT (resume=auto). log: $LOG"
exec python3 -m raiden.train \
  --config "$CONFIG" \
  --resume "${RAIDEN_RESUME:-auto}" \
  --train-path "$TRAIN_JSONL" \
  --val-path "$VAL_JSONL" \
  2>&1 | tee -a "$LOG"

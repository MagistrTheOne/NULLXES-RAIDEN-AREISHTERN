#!/usr/bin/env bash
# Download BF16 training weights onto the persistent volume.
# Product card remains zai-org/GLM-5.3-Flash; QLoRA needs unquantized Linears.

set -euo pipefail

if [[ -f /workspace/raiden.env ]]; then
  # shellcheck disable=SC1091
  source /workspace/raiden.env
fi

export RAIDEN_MODELS="${RAIDEN_MODELS:-/workspace/models}"
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
REPO="${RAIDEN_BASE_MODEL:-zai-org/GLM-5.3-Flash-BF16}"
DEST="${RAIDEN_MODELS}/GLM-5.3-Flash-BF16"

mkdir -p "$DEST"

if [[ -f "${DEST}/config.json" && -f "${DEST}/model.safetensors.index.json" ]]; then
  echo "model already present at ${DEST} — skip download"
else
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN not set; attempting public download"
  fi
  python3 -m pip install --break-system-packages -q "huggingface_hub[hf_transfer]" || true
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  python3 - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="${REPO}",
    local_dir="${DEST}",
    token=os.environ.get("HF_TOKEN"),
)
print("downloaded ${REPO} -> ${DEST}")
PY
fi

# Point training at the local snapshot so restarts never re-pull.
python3 - <<PY
from pathlib import Path
cfg = Path("${RAIDEN_ROOT:-/workspace/raiden}/configs/raiden_qlora.yaml")
if cfg.exists():
    text = cfg.read_text(encoding="utf-8")
    if "base_model: zai-org/GLM-5.3-Flash-BF16" in text:
        cfg.write_text(text.replace(
            "base_model: zai-org/GLM-5.3-Flash-BF16",
            "base_model: ${DEST}",
        ), encoding="utf-8")
        print("updated configs/raiden_qlora.yaml base_model -> ${DEST}")
PY

echo "download complete: ${DEST}"
ls -lh "${DEST}/config.json"

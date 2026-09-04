#!/usr/bin/env bash
# RunPod persistent-volume layout + Python deps.
# Intended image: official PyTorch template (system torch/CUDA already present).
# Do NOT create a fresh venv that drops the template torch.

set -euo pipefail

export RAIDEN_WORKSPACE="${RAIDEN_WORKSPACE:-/workspace}"
export RAIDEN_ROOT="${RAIDEN_ROOT:-${RAIDEN_WORKSPACE}/raiden}"
export RAIDEN_MODELS="${RAIDEN_MODELS:-${RAIDEN_WORKSPACE}/models}"
export RAIDEN_DATASETS="${RAIDEN_DATASETS:-${RAIDEN_WORKSPACE}/datasets}"
export RAIDEN_CHECKPOINTS="${RAIDEN_CHECKPOINTS:-${RAIDEN_WORKSPACE}/checkpoints}"
export RAIDEN_LOGS="${RAIDEN_LOGS:-${RAIDEN_WORKSPACE}/logs}"
export RAIDEN_CACHE="${RAIDEN_CACHE:-${RAIDEN_WORKSPACE}/cache}"
export HF_HOME="${HF_HOME:-${RAIDEN_CACHE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${RAIDEN_CACHE}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RAIDEN_CACHE}/triton}"
export TMPDIR="${TMPDIR:-${RAIDEN_CACHE}/tmp}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p \
  "$RAIDEN_ROOT" \
  "$RAIDEN_MODELS" \
  "$RAIDEN_DATASETS/raiden" \
  "$RAIDEN_CHECKPOINTS" \
  "$RAIDEN_LOGS" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TMPDIR"

# Persist this env across SSH sessions on the volume.
cat > "${RAIDEN_WORKSPACE}/raiden.env" <<EOF
export RAIDEN_WORKSPACE="${RAIDEN_WORKSPACE}"
export RAIDEN_ROOT="${RAIDEN_ROOT}"
export RAIDEN_MODELS="${RAIDEN_MODELS}"
export RAIDEN_DATASETS="${RAIDEN_DATASETS}"
export RAIDEN_CHECKPOINTS="${RAIDEN_CHECKPOINTS}"
export RAIDEN_LOGS="${RAIDEN_LOGS}"
export RAIDEN_CACHE="${RAIDEN_CACHE}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export TORCH_HOME="${TORCH_HOME}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export TMPDIR="${TMPDIR}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RAIDEN_ROOT}/src:\${PYTHONPATH:-}"
EOF

# shellcheck disable=SC1091
source "${RAIDEN_WORKSPACE}/raiden.env"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is set (${#HF_TOKEN} chars)."
else
  echo "WARNING: HF_TOKEN is empty. Gated downloads and Hub push will fail."
fi

REQ="${RAIDEN_ROOT}/requirements.txt"
if [[ ! -f "$REQ" ]]; then
  echo "ERROR: $REQ not found. Copy the raiden/ repo to ${RAIDEN_ROOT} first."
  exit 2
fi

python3 -m pip install --break-system-packages --upgrade pip
python3 -m pip install --break-system-packages -r "$REQ"

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
import transformers, peft, bitsandbytes, trl, accelerate
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("trl", trl.__version__)
print("accelerate", accelerate.__version__)
try:
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    print("glm5_next OK", Glm5NextConfig.model_type)
except Exception as e:
    print("glm5_next IMPORT FAILED:", e)
    raise SystemExit(3)
PY

echo "bootstrap complete"
echo "next: bash ${RAIDEN_ROOT}/scripts/download_model.sh"

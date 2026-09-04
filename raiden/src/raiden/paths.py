"""RunPod persistent-volume path layout.

Everything durable lives under RAIDEN_ROOT (default /workspace/raiden).
HF caches are redirected onto the same volume so pod restarts do not
re-download GLM-5.3-Flash.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def raiden_root() -> Path:
    return _env_path("RAIDEN_ROOT", "/workspace/raiden")


def workspace_root() -> Path:
    return _env_path("RAIDEN_WORKSPACE", "/workspace")


def models_dir() -> Path:
    return _env_path("RAIDEN_MODELS", str(workspace_root() / "models"))


def datasets_dir() -> Path:
    return _env_path("RAIDEN_DATASETS", str(workspace_root() / "datasets"))


def checkpoints_dir() -> Path:
    return _env_path("RAIDEN_CHECKPOINTS", str(workspace_root() / "checkpoints"))


def logs_dir() -> Path:
    return _env_path("RAIDEN_LOGS", str(workspace_root() / "logs"))


def cache_dir() -> Path:
    return _env_path("RAIDEN_CACHE", str(workspace_root() / "cache"))


def hf_home() -> Path:
    return _env_path("HF_HOME", str(cache_dir() / "huggingface"))


def apply_cache_env() -> dict[str, str]:
    """Force Hugging Face / Torch caches onto the persistent volume."""
    hf = str(hf_home())
    cache = str(cache_dir())
    mapping = {
        "HF_HOME": hf,
        "HUGGINGFACE_HUB_CACHE": str(Path(hf) / "hub"),
        "TRANSFORMERS_CACHE": str(Path(hf) / "transformers"),
        "HF_DATASETS_CACHE": str(Path(hf) / "datasets"),
        "TORCH_HOME": str(Path(cache) / "torch"),
        "XDG_CACHE_HOME": cache,
        "WANDB_DIR": str(logs_dir() / "wandb"),
        "TRITON_CACHE_DIR": str(Path(cache) / "triton"),
        "TMPDIR": str(Path(cache) / "tmp"),
    }
    for key, value in mapping.items():
        os.environ.setdefault(key, value)
        Path(value).mkdir(parents=True, exist_ok=True)
    for path in (
        raiden_root(),
        models_dir(),
        datasets_dir(),
        checkpoints_dir(),
        logs_dir(),
        cache_dir(),
        hf_home(),
    ):
        path.mkdir(parents=True, exist_ok=True)
    return mapping


def latest_checkpoint(run_dir: Path | None = None) -> Path | None:
    """Return the newest Hugging Face checkpoint-* directory, if any."""
    root = run_dir or (checkpoints_dir() / "raiden-sft-stage1")
    if not root.exists():
        return None
    ckpts = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if ckpts:
        return ckpts[-1]
    if (root / "adapter_config.json").exists() or (root / "adapter_model.safetensors").exists():
        return root
    return None

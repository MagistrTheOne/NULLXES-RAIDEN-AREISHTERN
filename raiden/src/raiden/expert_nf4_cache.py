"""Disk cache for packed-expert NF4.

Quantizing 42 MoE layers in-process during from_pretrained takes ~2–3 h on
one B300: each gate_up is ~9.7 GiB BF16 and bitsandbytes is invoked while
~274 GiB is already resident. The cache is the one-time cost. After that,
train load must not re-read those BF16 shards and must not quantize again.

This is still Stage I QLoRA. LoRA targets stay nn.Linear.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("raiden")

CACHE_MANIFEST = "manifest.json"
CACHE_VERSION = 1


def is_packed_expert_weight_key(key: str) -> bool:
    """True for Glm5Next packed expert Parameters, not shared-expert Linears."""
    parts = key.split(".")
    if "shared_experts" in parts:
        return False
    return key.endswith(".experts.gate_up_proj") or key.endswith(".experts.down_proj")


def expert_nf4_cache_dir(explicit: str | Path | None = None) -> Path | None:
    raw = explicit if explicit is not None else os.environ.get("RAIDEN_EXPERT_NF4_CACHE")
    if raw is None or str(raw).strip() == "":
        default = Path(os.environ.get("RAIDEN_CACHE", "/workspace/cache")) / "expert_nf4"
        if (default / CACHE_MANIFEST).is_file():
            return default
        return None
    return Path(str(raw)).expanduser()


def cache_file_for_key(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.pt"


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / CACHE_MANIFEST
    if not path.exists():
        return {"v": CACHE_VERSION, "tensors": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"v": CACHE_VERSION, "tensors": {}}
    if not isinstance(data, dict):
        return {"v": CACHE_VERSION, "tensors": {}}
    data.setdefault("v", CACHE_VERSION)
    data.setdefault("tensors", {})
    return data


def save_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_MANIFEST
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def peek_cached_shape(cache_dir: Path, key: str) -> tuple[int, ...] | None:
    man = load_manifest(cache_dir)
    info = (man.get("tensors") or {}).get(key)
    if not info:
        return None
    shape = info.get("shape")
    if not shape:
        return None
    return tuple(int(x) for x in shape)


def cache_has_tensor(cache_dir: Path, key: str) -> bool:
    if peek_cached_shape(cache_dir, key) is None:
        return False
    return cache_file_for_key(cache_dir, key).is_file()


def resolve_cached_key(cache_dir: Path, key: str) -> str | None:
    """HF module paths and safetensors keys sometimes differ by a `model.` prefix."""
    if cache_has_tensor(cache_dir, key):
        return key
    alts = []
    if key.startswith("model."):
        alts.append(key[len("model.") :])
    else:
        alts.append(f"model.{key}")
    if "." in key:
        alts.append(key.split(".", 1)[1])
    for alt in alts:
        if alt and cache_has_tensor(cache_dir, alt):
            return alt
    return None


def iter_index_weight_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(f"missing {index}")
    payload = json.loads(index.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index} has no weight_map")
    return {str(k): str(v) for k, v in weight_map.items()}


def packed_expert_keys(model_dir: Path) -> list[str]:
    keys = [k for k in iter_index_weight_map(model_dir) if is_packed_expert_weight_key(k)]
    keys.sort()
    return keys


def cache_missing_keys(model_dir: Path, cache_dir: Path) -> list[str]:
    return [k for k in packed_expert_keys(model_dir) if not cache_has_tensor(cache_dir, k)]


def cache_is_complete(model_dir: Path, cache_dir: Path) -> bool:
    if not cache_dir.is_dir():
        return False
    return not cache_missing_keys(model_dir, cache_dir)


def rows_to_blob(rows: list, shape: tuple[int, int, int]) -> dict[str, Any]:
    import torch

    if not rows:
        raise ValueError("empty NF4 rows")
    q = torch.cat([r[0].reshape(-1).detach().cpu() for r in rows])
    absmax = torch.cat([_row_absmax(r[1]).reshape(-1).detach().cpu() for r in rows])
    qs0 = rows[0][1]
    dtype = qs0.dtype
    dtype_name = str(dtype).replace("torch.", "") if dtype is not None else "bfloat16"
    return {
        "v": CACHE_VERSION,
        "shape": tuple(int(x) for x in shape),
        "blocksize": int(qs0.blocksize),
        "quant_type": qs0.quant_type,
        "dtype": dtype_name,
        "code": qs0.code.detach().cpu(),
        "q": q,
        "absmax": absmax.float(),
    }


def _row_absmax(qs):
    from raiden.expert_nf4 import _absmax_float

    return _absmax_float(qs)


def blob_to_rows(blob: dict[str, Any], device: str = "cuda") -> tuple[list, tuple[int, int, int]]:
    import torch
    from bitsandbytes.functional import QuantState

    from raiden.expert_nf4 import _split_packed_nf4

    shape = tuple(int(x) for x in blob["shape"])
    n_exp, out, inn = shape
    dtype_name = blob.get("dtype") or "bfloat16"
    dtype = getattr(torch, dtype_name, torch.bfloat16)
    code = blob["code"].to(device=device)
    q = blob["q"].to(device=device)
    absmax = blob["absmax"].to(device=device)
    qs = QuantState(
        absmax=absmax,
        shape=torch.Size((n_exp * out, inn)),
        code=code,
        blocksize=int(blob["blocksize"]),
        quant_type=blob["quant_type"],
        dtype=dtype,
    )
    return _split_packed_nf4(q, qs, n_exp, out, inn), shape


def save_expert_blob(cache_dir: Path, key: str, rows: list, shape: tuple[int, int, int]) -> Path:
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_file_for_key(cache_dir, key)
    tmp = path.with_suffix(".pt.tmp")
    blob = rows_to_blob(rows, shape)
    torch.save(blob, tmp)
    tmp.replace(path)
    man = load_manifest(cache_dir)
    man["tensors"][key] = {
        "shape": list(shape),
        "file": path.name,
        "bytes": path.stat().st_size,
    }
    save_manifest(cache_dir, man)
    logger.info("NF4 cache wrote %s shape=%s bytes=%s", key, shape, path.stat().st_size)
    return path


def load_expert_blob(cache_dir: Path, key: str, device: str = "cuda"):
    import torch

    path = cache_file_for_key(cache_dir, key)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob_to_rows(blob, device=device)


def apply_rows_to_module(module: Any, param_name: str, rows: list, shape: tuple[int, int, int]) -> int:
    store = getattr(module, "_raiden_nf4", None)
    if store is None:
        store = {}
        module._raiden_nf4 = store
    store[param_name] = rows
    if not hasattr(module, "_raiden_nf4_shape"):
        module._raiden_nf4_shape = {}
    module._raiden_nf4_shape[param_name] = shape
    return int(shape[0])

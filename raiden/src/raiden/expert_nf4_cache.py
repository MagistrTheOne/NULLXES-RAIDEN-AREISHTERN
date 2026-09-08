"""Disk cache for packed-expert NF4.

Quantizing 42 MoE layers in-process during from_pretrained takes ~2–3 h on
one B300: each gate_up is ~9.7 GiB BF16 and bitsandbytes is invoked while
~274 GiB is already resident. The cache is the one-time cost. After that,
train load must not re-read those BF16 shards and must not quantize again.

This is still Stage I QLoRA. LoRA targets stay nn.Linear.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("raiden")

CACHE_MANIFEST = "manifest.json"
CACHE_VERSION = 2
CACHE_STATES = ("MISSING", "MATERIALIZING", "INCOMPLETE", "READY", "CORRUPTED", "NO_INDEX")
BLOCK_TRAIN_STATES = frozenset({"MISSING", "MATERIALIZING", "INCOMPLETE", "CORRUPTED", "NO_INDEX"})


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
        return {"v": CACHE_VERSION, "tensors": {}, "status": "MISSING"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"v": CACHE_VERSION, "tensors": {}, "status": "CORRUPTED"}
    if not isinstance(data, dict):
        return {"v": CACHE_VERSION, "tensors": {}, "status": "CORRUPTED"}
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
    path = cache_file_for_key(cache_dir, key)
    if not path.is_file():
        return False
    info = (load_manifest(cache_dir).get("tensors") or {}).get(key) or {}
    nbytes = info.get("bytes")
    if nbytes is not None and int(nbytes) != path.stat().st_size:
        return False
    return True


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
    if cache_dir is None or not Path(cache_dir).is_dir():
        return False
    return inspect_cache_state(model_dir, cache_dir) == "READY"


def cache_dir_stats(cache_dir: Path) -> dict[str, int]:
    n_files = 0
    n_bytes = 0
    if cache_dir.is_dir():
        for p in cache_dir.rglob("*"):
            if p.is_file():
                n_files += 1
                n_bytes += p.stat().st_size
    return {"n_files": n_files, "n_bytes": n_bytes}


def estimated_nf4_cache_bytes(n_moe_layers: int) -> int:
    """NF4 q-payload only (no torch.save overhead). gate_up 288x4096x4096 + down 288x4096x2048."""
    gate_q = 288 * 4096 * 4096 // 2
    down_q = 288 * 4096 * 2048 // 2
    return int(n_moe_layers) * (gate_q + down_q)


def manifest_fingerprint(tensors: dict) -> str:
    lines = []
    for key in sorted(tensors):
        info = tensors[key] or {}
        shape = tuple(info.get("shape") or ())
        lines.append(f"{key}\t{info.get('bytes')}\t{shape}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def live_expert_quant_allowed() -> bool:
    return os.environ.get("RAIDEN_ALLOW_LIVE_EXPERT_QUANT", "").strip().lower() in {"1", "true", "yes"}


def refuse_unready_expert_cache(report: dict[str, Any]) -> None:
    """Train may start only on cache=READY. A folder is not a verified cache."""
    from raiden.compatibility import RaidenCompatibilityError

    status = str(report.get("cache") or "MISSING")
    if status == "READY":
        return
    if live_expert_quant_allowed():
        logger.warning(
            "expert cache=%s; RAIDEN_ALLOW_LIVE_EXPERT_QUANT=1 — continuing with LIVE_QUANT",
            status,
        )
        return
    hint = {
        "MISSING": "run python -m raiden.materialize_experts",
        "MATERIALIZING": "materialize is running or crashed before finalize; wait or re-run materialize_experts",
        "INCOMPLETE": "re-run python -m raiden.materialize_experts (resume-safe); do not start SFT at 97%",
        "CORRUPTED": "size or checksum mismatch; re-run materialize_experts to rewrite bad keys",
        "NO_INDEX": "base_model has no model.safetensors.index.json",
    }.get(status, "fix the expert NF4 cache")
    raise RaidenCompatibilityError(
        f"packed-expert NF4 cache is {status} "
        f"({report.get('cached_experts')}/{report.get('expected_experts')}). "
        f"{hint}. cache=READY means the manifest was verified, not that the folder exists. "
        "Do not mix LIVE_QUANT with a partial cache."
    )


def begin_materialize(model_dir: Path, cache_dir: Path) -> dict[str, Any]:
    keys = packed_expert_keys(model_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    man = load_manifest(cache_dir)
    if man.get("status") == "CORRUPTED" and not man.get("tensors"):
        man = {"v": CACHE_VERSION, "tensors": {}}
    man.update(
        {
            "v": CACHE_VERSION,
            "model": Path(model_dir).name,
            "quant": "nf4",
            "status": "MATERIALIZING",
            "expected_keys": keys,
            "layers": sum(1 for k in keys if k.endswith("gate_up_proj")),
            "experts_per_layer": man.get("experts_per_layer"),
            "checksum": None,
        }
    )
    man.setdefault("tensors", {})
    save_manifest(cache_dir, man)
    logger.info(
        "materialize status=MATERIALIZING model=%s layers=%s expected_keys=%s",
        man["model"],
        man["layers"],
        len(keys),
    )
    return man


def finalize_materialize(model_dir: Path, cache_dir: Path) -> dict[str, Any]:
    keys = packed_expert_keys(model_dir)
    man = load_manifest(cache_dir)
    missing = [k for k in keys if not cache_has_tensor(cache_dir, k)]
    if missing:
        man["status"] = "INCOMPLETE"
        man["checksum"] = None
        save_manifest(cache_dir, man)
        return man
    tensors = man.get("tensors") or {}
    first = next(iter(tensors.values()), None)
    shape = (first or {}).get("shape") or []
    man["experts_per_layer"] = int(shape[0]) if shape else None
    man["layers"] = sum(1 for k in keys if k.endswith("gate_up_proj"))
    man["checksum"] = manifest_fingerprint(tensors)
    man["status"] = "READY"
    save_manifest(cache_dir, man)
    logger.info(
        "materialize status=READY layers=%s experts_per_layer=%s checksum=%s",
        man["layers"],
        man["experts_per_layer"],
        man["checksum"][:12],
    )
    return man


def inspect_cache_state(model_dir: Path, cache_dir: Path | None) -> str:
    """Return MISSING | MATERIALIZING | INCOMPLETE | READY | CORRUPTED | NO_INDEX.

    READY is a verified manifest, not 'the folder exists'. A 97% crash stays
    MATERIALIZING or INCOMPLETE until finalize_materialize writes READY.
    """
    if cache_dir is None:
        return "MISSING"
    cache_dir = Path(cache_dir)
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.exists():
        return "NO_INDEX"
    try:
        expected = packed_expert_keys(Path(model_dir))
    except Exception:
        return "NO_INDEX"
    man_path = cache_dir / CACHE_MANIFEST
    if not man_path.exists():
        if cache_dir.is_dir() and any(cache_dir.glob("*.pt")):
            return "INCOMPLETE"
        return "MISSING"
    man = load_manifest(cache_dir)
    if man.get("status") == "CORRUPTED" and not (man.get("tensors") or {}):
        return "CORRUPTED"
    tensors = man.get("tensors") or {}
    present = 0
    for key in expected:
        info = tensors.get(key)
        path = cache_file_for_key(cache_dir, key)
        if info is None or not path.is_file():
            continue
        nbytes = info.get("bytes")
        if nbytes is not None and int(nbytes) != path.stat().st_size:
            return "CORRUPTED"
        present += 1
    declared = str(man.get("status") or "")
    if declared == "MATERIALIZING":
        return "MATERIALIZING"
    if present == 0:
        return "CORRUPTED" if declared == "READY" else "MISSING"
    if present < len(expected):
        return "CORRUPTED" if declared == "READY" else "INCOMPLETE"
    checksum = man.get("checksum")
    if checksum and checksum != manifest_fingerprint(tensors):
        return "CORRUPTED"
    if declared == "READY" or present == len(expected):
        return "READY"
    return "INCOMPLETE"


def expert_cache_preflight(model_dir: str | Path, cache_dir: Path | None, policy: str) -> dict[str, Any]:
    """Black-box line before from_pretrained. READY means verified, not 'folder exists'."""
    model_dir = Path(model_dir)
    status = inspect_cache_state(model_dir, cache_dir)
    expected = 0
    cached = 0
    layers = None
    experts_per_layer = None
    checksum = ""
    try:
        keys = packed_expert_keys(model_dir)
        expected = len(keys)
        if cache_dir is not None:
            cached = sum(1 for k in keys if cache_has_tensor(cache_dir, k))
            man = load_manifest(cache_dir)
            layers = man.get("layers")
            experts_per_layer = man.get("experts_per_layer")
            checksum = str(man.get("checksum") or "")
    except FileNotFoundError:
        status = "NO_INDEX"
    stats = cache_dir_stats(cache_dir) if cache_dir is not None else {"n_files": 0, "n_bytes": 0}
    bf16_read = "SKIPPED" if status == "READY" else "LIVE_QUANT"
    report = {
        "expert_policy": policy,
        "cache": status,
        "cache_dir": str(cache_dir) if cache_dir is not None else "",
        "cached_experts": cached,
        "expected_experts": expected,
        "layers": layers,
        "experts_per_layer": experts_per_layer,
        "checksum": checksum,
        "bf16_expert_read": bf16_read,
        "cache_files": stats["n_files"],
        "cache_bytes": stats["n_bytes"],
    }
    logger.info(
        "expert_policy=%s cache=%s cached_experts=%s/%s layers=%s experts_per_layer=%s "
        "bf16_expert_read=%s checksum=%s cache_dir=%s files=%s bytes=%s",
        report["expert_policy"],
        report["cache"],
        report["cached_experts"],
        report["expected_experts"],
        report["layers"],
        report["experts_per_layer"],
        report["bf16_expert_read"],
        (checksum[:12] + "…") if checksum else "(none)",
        report["cache_dir"] or "(unset)",
        report["cache_files"],
        report["cache_bytes"],
    )
    return report


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
    if man.get("status") != "MATERIALIZING":
        man["status"] = "MATERIALIZING"
    if shape:
        man["experts_per_layer"] = int(shape[0])
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

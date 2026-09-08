"""One-time packed-expert NF4 materialize. Run on the pod with an empty GPU.

Does not load the 643 GiB model through from_pretrained.

Official GLM-5.3-Flash-BF16 is Linear-on-disk, packed-at-runtime:
stack 288 gate/up/down Linears per layer into gate_up_proj / down_proj,
quantize, write two cache blobs per MoE layer. Resume-safe.

    python -m raiden.materialize_experts \\
      --model /workspace/models/GLM-5.3-Flash-BF16 \\
      --out /workspace/cache/expert_nf4
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

from raiden.expert_nf4 import quantize_packed_expert_tensor
from raiden.expert_nf4_cache import (
    moe_layer_sort_key,
    begin_materialize,
    cache_dir_stats,
    cache_has_tensor,
    cache_missing_keys,
    expert_storage_layout,
    finalize_materialize,
    fuse_gate_up_stacked,
    group_linear_expert_keys,
    iter_index_weight_map,
    load_manifest,
    packed_expert_keys,
    record_linear_shapes,
    runtime_expert_layout,
    save_expert_blob,
)
from raiden.logging import setup_logging
from raiden.paths import apply_cache_env


def _load_one_tensor(model_dir: Path, key: str, shard_name: str):
    from safetensors import safe_open

    shard = model_dir / shard_name
    with safe_open(str(shard), framework="pt", device="cpu") as fh:
        return fh.get_tensor(key)


def _load_stacked(model_dir: Path, weight_map: dict[str, str], id_to_key: dict[int, str]):
    from safetensors import safe_open
    import torch

    ids = sorted(id_to_key)
    if ids != list(range(len(ids))):
        raise SystemExit(
            f"expert ids are not 0..n-1 (n={len(ids)} first={ids[:3]} last={ids[-3:]})"
        )
    by_shard: dict[str, list[tuple[int, str]]] = {}
    for eid in ids:
        key = id_to_key[eid]
        by_shard.setdefault(weight_map[key], []).append((eid, key))
    found: dict[int, object] = {}
    for shard, items in by_shard.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as fh:
            for eid, key in items:
                found[eid] = fh.get_tensor(key)
    missing = [i for i in ids if i not in found]
    if missing:
        raise SystemExit(f"failed to load expert ids {missing[:8]}")
    stacked = torch.stack([found[i] for i in ids], dim=0)
    del found
    return stacked


def _materialize_packed_3d(model_dir: Path, cache_dir: Path, keys: list[str], weight_map: dict[str, str], log) -> None:
    import torch

    for i, key in enumerate(keys, 1):
        if cache_has_tensor(cache_dir, key):
            log.info("skip %s/%s %s (cache hit)", i, len(keys), key)
            continue
        shard = weight_map[key]
        log.info("read %s/%s %s from %s", i, len(keys), key, shard)
        tensor = _load_one_tensor(model_dir, key, shard)
        if getattr(tensor, "ndim", 0) != 3:
            log.warning("skip %s: not a packed 3D Parameter (ndim=%s)", key, getattr(tensor, "ndim", None))
            del tensor
            continue
        rows, shape = quantize_packed_expert_tensor(tensor, log_name=key)
        del tensor
        save_expert_blob(cache_dir, key, rows, shape)
        del rows
        gc.collect()
        torch.cuda.empty_cache()
        log.info("wrote %s/%s %s shape=%s", i, len(keys), key, shape)


def _materialize_from_linear(model_dir: Path, cache_dir: Path, weight_map: dict[str, str], log) -> None:
    import torch

    groups = group_linear_expert_keys(weight_map)
    prefixes = sorted(groups, key=moe_layer_sort_key)
    n = len(prefixes)
    log.info(
        "stacking per-expert Linears into packed runtime NF4 layers=%s (2 blobs/layer, not 37152 files)",
        n,
    )
    linear_shapes: dict[str, list[int]] = dict(load_manifest(cache_dir).get("linear_shapes") or {})
    for i, prefix in enumerate(prefixes, 1):
        gu_key = f"{prefix}.gate_up_proj"
        dn_key = f"{prefix}.down_proj"
        if cache_has_tensor(cache_dir, gu_key) and cache_has_tensor(cache_dir, dn_key):
            log.info("skip %s/%s %s (cache hit)", i, n, prefix)
            continue
        g = groups[prefix]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if not g[proj]:
                raise SystemExit(f"{prefix} missing {proj} in checkpoint index")
        n_exp = len(g["gate_proj"])
        if len(g["up_proj"]) != n_exp or len(g["down_proj"]) != n_exp:
            raise SystemExit(
                f"{prefix} expert count mismatch gate={len(g['gate_proj'])} "
                f"up={len(g['up_proj'])} down={len(g['down_proj'])}"
            )
        log.info("stack %s/%s %s experts=%s", i, n, prefix, n_exp)
        if not cache_has_tensor(cache_dir, gu_key):
            stacked_gate = _load_stacked(model_dir, weight_map, g["gate_proj"])
            stacked_up = _load_stacked(model_dir, weight_map, g["up_proj"])
            linear_shapes["gate_proj"] = [int(x) for x in stacked_gate.shape[1:]]
            linear_shapes["up_proj"] = [int(x) for x in stacked_up.shape[1:]]
            gate_up = fuse_gate_up_stacked(stacked_gate, stacked_up)
            del stacked_gate, stacked_up
            gc.collect()
            rows, shape = quantize_packed_expert_tensor(gate_up, log_name=gu_key)
            del gate_up
            save_expert_blob(cache_dir, gu_key, rows, shape)
            del rows
            gc.collect()
            torch.cuda.empty_cache()
            log.info("wrote %s/%s %s shape=%s", i, n, gu_key, shape)
        if not cache_has_tensor(cache_dir, dn_key):
            stacked_down = _load_stacked(model_dir, weight_map, g["down_proj"])
            linear_shapes["down_proj"] = [int(x) for x in stacked_down.shape[1:]]
            rows, shape = quantize_packed_expert_tensor(stacked_down, log_name=dn_key)
            del stacked_down
            save_expert_blob(cache_dir, dn_key, rows, shape)
            del rows
            gc.collect()
            torch.cuda.empty_cache()
            log.info("wrote %s/%s %s shape=%s", i, n, dn_key, shape)
    if linear_shapes:
        record_linear_shapes(cache_dir, linear_shapes)


def materialize(model_dir: Path, cache_dir: Path) -> int:
    import logging

    import torch

    log = logging.getLogger("raiden")
    if not torch.cuda.is_available():
        raise SystemExit("materialize_experts needs CUDA on the training pod")
    from raiden.expert_nf4_cache import resolve_model_dir

    model_dir = resolve_model_dir(model_dir)
    layout = expert_storage_layout(model_dir)
    runtime = runtime_expert_layout(model_dir)
    keys = packed_expert_keys(model_dir)
    if not keys:
        raise SystemExit(f"no packed expert keys in {model_dir}")
    weight_map = iter_index_weight_map(model_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    begin_materialize(model_dir, cache_dir)
    missing = [k for k in keys if not cache_has_tensor(cache_dir, k)]
    log.info(
        "materialize packed-expert NF4 model=%s out=%s checkpoint_layout=%s "
        "expert_layout=%s total=%s missing=%s gpu=%s",
        model_dir,
        cache_dir,
        layout,
        runtime,
        len(keys),
        len(missing),
        torch.cuda.get_device_name(0),
    )
    if layout == "per_expert_linear":
        _materialize_from_linear(model_dir, cache_dir, weight_map, log)
    else:
        _materialize_packed_3d(model_dir, cache_dir, keys, weight_map, log)
    leftover = cache_missing_keys(model_dir, cache_dir)
    man = finalize_materialize(model_dir, cache_dir)
    stats = cache_dir_stats(cache_dir)
    if leftover or man.get("status") != "READY":
        log.error(
            "materialize %s, missing %s keys (first=%s)",
            man.get("status") or "INCOMPLETE",
            len(leftover),
            leftover[0] if leftover else "(none)",
        )
        log.info("cache files=%s bytes=%s", stats["n_files"], stats["n_bytes"])
        return 2
    log.info(
        "materialize complete status=READY tensors=%s layers=%s experts_per_layer=%s "
        "checksum=%s dir=%s files=%s bytes=%s (~%.1f GiB)",
        man.get("layers", 0) * 2,
        man.get("layers"),
        man.get("experts_per_layer"),
        man.get("checksum"),
        cache_dir,
        stats["n_files"],
        stats["n_bytes"],
        stats["n_bytes"] / 1024**3,
    )
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Materialize packed-expert NF4 cache")
    p.add_argument(
        "--model",
        default=os.environ.get("RAIDEN_MODEL_DIR", "/workspace/models/GLM-5.3-Flash-BF16"),
    )
    p.add_argument(
        "--out",
        default=os.environ.get("RAIDEN_EXPERT_NF4_CACHE", "/workspace/cache/expert_nf4"),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    apply_cache_env()
    args = parse_args(argv)
    os.environ["RAIDEN_EXPERT_NF4_CACHE"] = str(Path(args.out))
    setup_logging("raiden-materialize")
    return materialize(Path(args.model), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())

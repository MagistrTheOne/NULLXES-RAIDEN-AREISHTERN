"""One-time packed-expert NF4 materialize. Run on the pod with an empty GPU.

Does not load the 643 GiB model through from_pretrained. Walks safetensors
shards, quantizes only packed expert Parameters, writes
RAIDEN_EXPERT_NF4_CACHE. Resume-safe: finished keys are skipped.

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
    begin_materialize,
    cache_dir_stats,
    cache_has_tensor,
    cache_missing_keys,
    finalize_materialize,
    iter_index_weight_map,
    packed_expert_keys,
    save_expert_blob,
)
from raiden.logging import setup_logging
from raiden.paths import apply_cache_env


def _load_one_tensor(model_dir: Path, key: str, shard_name: str):
    from safetensors import safe_open

    shard = model_dir / shard_name
    with safe_open(str(shard), framework="pt", device="cpu") as fh:
        return fh.get_tensor(key)


def materialize(model_dir: Path, cache_dir: Path) -> int:
    import logging

    import torch

    log = logging.getLogger("raiden")
    if not torch.cuda.is_available():
        raise SystemExit("materialize_experts needs CUDA on the training pod")
    keys = packed_expert_keys(model_dir)
    if not keys:
        raise SystemExit(f"no packed expert keys in {model_dir}")
    weight_map = iter_index_weight_map(model_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    begin_materialize(model_dir, cache_dir)
    missing = [k for k in keys if not cache_has_tensor(cache_dir, k)]
    log.info(
        "materialize packed-expert NF4 model=%s out=%s total=%s missing=%s gpu=%s",
        model_dir,
        cache_dir,
        len(keys),
        len(missing),
        torch.cuda.get_device_name(0),
    )
    done = len(keys) - len(missing)
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
        done += 1
        log.info("wrote %s/%s %s shape=%s", i, len(keys), key, shape)
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
        done,
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

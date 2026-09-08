"""Dry-run before SFT. Folder existence is not a green light.

Default: verify the packed-expert NF4 manifest (seconds).
--full: load the QLoRA stack and assert freeze / LoRA / cache READY.

    python -m raiden.validate_runtime --config configs/raiden_qlora.yaml
    python -m raiden.validate_runtime --config configs/raiden_qlora.yaml --full
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from raiden.compatibility import RaidenCompatibilityError
from raiden.config import RaidenConfig
from raiden.expert_nf4 import count_nf4_expert_modules, resolve_packed_expert_policy
from raiden.expert_nf4_cache import (
    expert_cache_preflight,
    expert_nf4_cache_dir,
    expert_storage_layout,
    refuse_unready_expert_cache,
)
from raiden.freeze import (
    assert_lora_attached,
    assert_packed_experts_frozen,
    assert_router_frozen,
    assert_vision_frozen,
)
from raiden.logging import setup_logging
from raiden.paths import apply_cache_env

logger = logging.getLogger("raiden")

Check = tuple[str, bool, str]


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def cache_checks(model_dir: str, policy: str) -> tuple[list[Check], dict[str, Any]]:
    cache_dir = expert_nf4_cache_dir()
    pre = expert_cache_preflight(model_dir, cache_dir, policy)
    layout = pre.get("layout") or expert_storage_layout(model_dir)
    if layout == "per_expert_linear" or pre["cache"] == "NOT_REQUIRED":
        checks = [
            (
                "expert layout per-expert Linear",
                True,
                "BnB Linear4bit; packed NF4 cache not used",
            ),
            (
                "BF16 experts go through bitsandbytes",
                pre["bf16_expert_read"] == "BNB_LINEAR4BIT",
                f"bf16_expert_read={pre['bf16_expert_read']}",
            ),
        ]
        return checks, pre
    checks = [
        (
            "expert cache READY",
            pre["cache"] == "READY",
            f"cache={pre['cache']} {pre['cached_experts']}/{pre['expected_experts']}",
        ),
        (
            "BF16 expert path disabled",
            pre["bf16_expert_read"] == "SKIPPED",
            f"bf16_expert_read={pre['bf16_expert_read']}",
        ),
    ]
    return checks, pre


def _try(label: str, fn) -> Check:
    try:
        fn()
        return (label, True, "ok")
    except Exception as exc:  # noqa: BLE001
        return (label, False, str(exc).split("\n", 1)[0])


def loaded_model_checks(model: Any, pre: dict[str, Any], freeze_cfg) -> list[Check]:
    n_nf4 = count_nf4_expert_modules(model)
    layout = pre.get("layout") or ""
    checks: list[Check] = [
        ("base model loaded", model is not None, type(model).__name__),
        _try("router frozen", lambda: assert_router_frozen(model) if freeze_cfg.freeze_router else None),
        _try("vision frozen", lambda: assert_vision_frozen(model) if freeze_cfg.freeze_vision else None),
        _try(
            "experts frozen",
            lambda: assert_packed_experts_frozen(model) if freeze_cfg.freeze_packed_experts else None,
        ),
        _try("LoRA attached", lambda: assert_lora_attached(model)),
    ]
    if layout == "per_expert_linear" or pre.get("cache") == "NOT_REQUIRED":
        checks.extend(
            [
                ("expert cache not required", True, f"layout={layout or 'per_expert_linear'}"),
                (
                    "BF16 experts go through bitsandbytes",
                    pre.get("bf16_expert_read") == "BNB_LINEAR4BIT",
                    f"bf16_expert_read={pre.get('bf16_expert_read')}",
                ),
            ]
        )
    else:
        checks.extend(
            [
                (
                    "expert cache READY",
                    pre.get("cache") == "READY",
                    f"cache={pre.get('cache')}",
                ),
                (
                    "BF16 expert path disabled",
                    pre.get("bf16_expert_read") == "SKIPPED",
                    f"bf16_expert_read={pre.get('bf16_expert_read')}",
                ),
                ("NF4 experts attached", n_nf4 > 0, f"nf4_modules={n_nf4}"),
            ]
        )
    return checks


def print_checks(checks: list[Check]) -> bool:
    ok = True
    for label, passed, detail in checks:
        ok = ok and passed
        logger.info("%s %s — %s", _mark(passed), label, detail)
        print(f"{_mark(passed)} {label} — {detail}", flush=True)
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RAIDEN runtime dry-run before SFT")
    p.add_argument("--config", default="configs/raiden_qlora.yaml")
    p.add_argument(
        "--full",
        action="store_true",
        help="Load the QLoRA model and assert freeze/LoRA (GPU, minutes). Default is cache-only.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    apply_cache_env()
    args = parse_args(argv)
    setup_logging("raiden-validate")
    cfg = RaidenConfig.from_yaml(args.config)
    policy = resolve_packed_expert_policy(cfg.qlora.packed_expert_policy)
    print("RAIDEN runtime validate", flush=True)
    checks, pre = cache_checks(cfg.base_model, policy)
    if not args.full:
        ok = print_checks(checks)
        if not ok:
            try:
                refuse_unready_expert_cache(pre)
            except RaidenCompatibilityError as exc:
                logger.error("%s", exc)
                print(exc, file=sys.stderr)
            return 3
        print("runtime READY (cache). Pass --full to load the model before SFT.", flush=True)
        return 0

    try:
        refuse_unready_expert_cache(pre)
        from raiden.model import load_qlora_model

        model, _plan, _census = load_qlora_model(cfg)
        checks = loaded_model_checks(model, pre, cfg.freeze)
    except RaidenCompatibilityError as exc:
        logger.error("%s", exc)
        print(f"✗ load aborted — {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        logger.exception("validate --full failed")
        print(f"✗ load aborted — {exc}", file=sys.stderr)
        return 1
    ok = print_checks(checks)
    if not ok:
        return 3
    print("runtime READY. SFT may start.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

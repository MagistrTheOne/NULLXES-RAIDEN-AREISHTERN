"""Freeze policy for Stage I: router, vision, packed experts, indexer, mHC."""

from __future__ import annotations

from typing import Any

from raiden.lora import (
    EXPERT_PATH,
    HC_PATH,
    INDEXER_PATH,
    ROUTER_PATH,
    VISION_PATH,
    classify_path,
)


def freeze_module(mod: Any) -> None:
    for p in mod.parameters():
        p.requires_grad = False


def apply_freeze(model: Any, freeze_cfg) -> dict[str, int]:
    """Force-disable grads on protected subtrees even if PEFT wrapped them."""
    n_frozen = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        kind = classify_path(name)
        should_freeze = False
        if freeze_cfg.freeze_vision and (kind == "vision" or VISION_PATH.search(name)):
            should_freeze = True
        if freeze_cfg.freeze_router and not (
            freeze_cfg.router_experiment_enabled and freeze_cfg.router_experiment_train_router
        ):
            if kind == "router" or (ROUTER_PATH.search(name) and "gate_proj" not in name):
                should_freeze = True
        if freeze_cfg.freeze_packed_experts and (kind == "expert" or EXPERT_PATH.search(name)):
            should_freeze = True
        if freeze_cfg.freeze_indexer and (kind == "indexer" or INDEXER_PATH.search(name)):
            should_freeze = True
        if freeze_cfg.freeze_hyper_connections and (kind == "hyper_connection" or HC_PATH.search(name)):
            should_freeze = True
        if freeze_cfg.freeze_embeddings and "embed_tokens" in name:
            should_freeze = True
        if freeze_cfg.freeze_lm_head and "lm_head" in name and "lora_" not in name:
            should_freeze = True
        # LoRA params must stay trainable.
        if "lora_" in name:
            should_freeze = False
        if should_freeze:
            param.requires_grad = False
            n_frozen += param.numel()
        elif param.requires_grad:
            n_trainable += param.numel()
    visual = _find_visual(model)
    if freeze_cfg.freeze_vision and visual is not None:
        freeze_module(visual)
    return {"frozen_params": n_frozen, "trainable_params": n_trainable}


def _find_visual(model: Any):
    for attr in ("visual", "model"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        if hasattr(obj, "visual"):
            return obj.visual
        if type(obj).__name__.lower().startswith("glm5nextvision"):
            return obj
    base = getattr(model, "get_base_model", None)
    if callable(base):
        inner = base()
        if hasattr(inner, "visual"):
            return inner.visual
        if hasattr(inner, "model") and hasattr(inner.model, "visual"):
            return inner.model.visual
    return None


def assert_router_frozen(model: Any) -> None:
    leaked = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            continue
        if ROUTER_PATH.search(name) and "gate_proj" not in name:
            leaked.append(name)
    if leaked:
        raise RuntimeError(
            "Stage I router freeze violated. Trainable router tensors:\n  "
            + "\n  ".join(leaked[:20])
        )

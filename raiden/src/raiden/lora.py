"""LoRA target discovery for Glm5Next — no Llama-name assumptions.

Stage I policy:
  INCLUDE nn.Linear in language-model attention + dense/shared MLP.
  EXCLUDE vision, router/gate, packed experts, indexer, hyper-connections,
          embeddings, lm_head.

Router experiment is wired but disabled.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    import torch.nn as nn
except Exception:  # pragma: no cover
    nn = None  # type: ignore


# Names that MAY exist on Glm5Next after named_modules() introspection.
# These are candidates, not a hard-coded Llama list. Discovery still filters
# by actual nn.Linear instances present on the loaded model.
STAGE1_PREFERRED_LINEAR_NAMES = (
    # linear attention (KDA)
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "qkv_proj",
    "fused_qkvbfg_a_proj",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "b_proj",
    # sparse MLA
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    # dense MLP + shared expert
    "gate_proj",
    "up_proj",
    "down_proj",
)

INDEXER_LINEAR_NAMES = ("wq_b", "wk", "weights_proj")
VISION_LINEAR_NAMES = ("qkv", "proj")

ROUTER_PATH = re.compile(
    r"(?:^|\.)(?:gate|router)(?:$|\.)|e_score_correction_bias|(?:^|\.)mlp\.gate(?:$|\.)",
    re.I,
)
VISION_PATH = re.compile(r"(?:^|\.)(?:visual|vision|vision_model|vision_tower)(?:$|\.)", re.I)
EXPERT_PATH = re.compile(r"(?:^|\.)experts(?:$|\.)", re.I)
INDEXER_PATH = re.compile(r"(?:^|\.)indexer(?:$|\.)", re.I)
HC_PATH = re.compile(r"(?:hc_attn|hc_ffn|hyper_connection|hyper_head)", re.I)
EMBED_PATH = re.compile(r"embed_tokens|embed_in|word_embeddings", re.I)
HEAD_PATH = re.compile(r"(?:^|\.)lm_head(?:$|\.)", re.I)

ROUTER_EXPERIMENT_NOTE = (
    "RAIDEN ROUTER EXPERIMENT is implemented as a config flag and is DISABLED "
    "by default. Stage I freezes Glm5NextTextTopkRouter (mlp.gate.weight). "
    "Do not enable until identity SFT is proven without routing drift."
)


@dataclass
class ModuleCensus:
    architecture: str | None
    n_modules: int
    linear_modules: list[str]
    conv1d_modules: list[str]
    parameter_only_modules: list[str]
    router_modules: list[str]
    expert_modules: list[str]
    vision_modules: list[str]
    indexer_modules: list[str]
    hc_modules: list[str]
    param_count_by_kind: dict[str, int]
    n_layers: int | None = None
    layer_types: list[str] = field(default_factory=list)
    mlp_layer_types: list[str] = field(default_factory=list)


@dataclass
class LoRAPlan:
    target_modules: list[str]
    target_full_paths: list[str]
    excluded_full_paths: list[str]
    modules_to_save: list[str]
    freeze_patterns: list[str]
    notes: list[str]


def _is_linear(mod: Any) -> bool:
    if nn is None:
        return type(mod).__name__ in {"Linear", "Linear4bit", "Params4bit"}
    return isinstance(mod, nn.Linear) or type(mod).__name__ in {"Linear4bit", "Params4bit"}


def _is_conv1d(mod: Any) -> bool:
    if nn is None:
        return "Conv1d" in type(mod).__name__
    return isinstance(mod, nn.Conv1d) or "Conv1d" in type(mod).__name__


def _nparams(mod: Any) -> int:
    try:
        return sum(p.numel() for p in mod.parameters(recurse=False))
    except Exception:
        return 0


def classify_path(name: str) -> str | None:
    if VISION_PATH.search(name):
        return "vision"
    if EXPERT_PATH.search(name):
        return "expert"
    if ROUTER_PATH.search(name) and "gate_proj" not in name.split(".")[-1]:
        return "router"
    if INDEXER_PATH.search(name):
        return "indexer"
    if HC_PATH.search(name):
        return "hyper_connection"
    if EMBED_PATH.search(name):
        return "embedding"
    if HEAD_PATH.search(name):
        return "lm_head"
    return None


def census_from_model(model: Any) -> ModuleCensus:
    linear, conv, param_only = [], [], []
    router, expert, vision, indexer, hc = [], [], [], [], []
    counts = {
        "linear": 0,
        "conv1d": 0,
        "router": 0,
        "expert": 0,
        "vision": 0,
        "indexer": 0,
        "hyper_connection": 0,
        "other": 0,
    }
    n = 0
    for name, mod in model.named_modules():
        n += 1
        kind = classify_path(name)
        np = _nparams(mod)
        if kind == "vision":
            vision.append(name)
            counts["vision"] += np
        elif kind == "expert":
            expert.append(name)
            counts["expert"] += np
        elif kind == "router":
            router.append(name)
            counts["router"] += np
        elif kind == "indexer":
            indexer.append(name)
            counts["indexer"] += np
        elif kind == "hyper_connection":
            hc.append(name)
            counts["hyper_connection"] += np
        if _is_linear(mod):
            linear.append(name)
            if kind is None:
                counts["linear"] += np
        elif _is_conv1d(mod):
            conv.append(name)
            counts["conv1d"] += np
        elif kind is None:
            counts["other"] += np
        if kind == "expert" and not _is_linear(mod) and np:
            param_only.append(name)

    cfg = getattr(model, "config", None)
    text = getattr(cfg, "text_config", cfg) if cfg is not None else None
    arch = None
    n_layers = None
    layer_types: list[str] = []
    mlp_types: list[str] = []
    if cfg is not None:
        archs = getattr(cfg, "architectures", None)
        arch = archs[0] if archs else getattr(cfg, "model_type", None)
    if text is not None:
        n_layers = getattr(text, "num_hidden_layers", None)
        layer_types = list(getattr(text, "layer_types", []) or [])
        mlp_types = list(getattr(text, "mlp_layer_types", []) or [])

    return ModuleCensus(
        architecture=str(arch) if arch else None,
        n_modules=n,
        linear_modules=linear,
        conv1d_modules=conv,
        parameter_only_modules=param_only,
        router_modules=router,
        expert_modules=expert,
        vision_modules=vision,
        indexer_modules=indexer,
        hc_modules=hc,
        param_count_by_kind=counts,
        n_layers=n_layers,
        layer_types=layer_types,
        mlp_layer_types=mlp_types,
    )


def plan_stage1_lora(
    model: Any,
    *,
    freeze_router: bool = True,
    freeze_vision: bool = True,
    freeze_packed_experts: bool = True,
    freeze_indexer: bool = True,
    freeze_hyper_connections: bool = True,
    freeze_embeddings: bool = True,
    freeze_lm_head: bool = True,
    router_experiment_enabled: bool = False,
    router_experiment_train_router: bool = False,
    target_mode: str = "auto_stage1",
    explicit_targets: Iterable[str] = (),
) -> LoRAPlan:
    notes = [ROUTER_EXPERIMENT_NOTE]
    include_paths: list[str] = []
    exclude_paths: list[str] = []
    freeze_patterns = []

    if freeze_vision:
        freeze_patterns.append("visual")
    if freeze_router and not (router_experiment_enabled and router_experiment_train_router):
        freeze_patterns.append("mlp.gate")
    if freeze_packed_experts:
        freeze_patterns.append("mlp.experts")
    if freeze_indexer:
        freeze_patterns.append("indexer")
    if freeze_hyper_connections:
        freeze_patterns.append("hc_")
    if freeze_embeddings:
        freeze_patterns.append("embed_tokens")
    if freeze_lm_head:
        freeze_patterns.append("lm_head")

    preferred = set(STAGE1_PREFERRED_LINEAR_NAMES)
    if not freeze_indexer:
        preferred.update(INDEXER_LINEAR_NAMES)

    for name, mod in model.named_modules():
        if not _is_linear(mod):
            continue
        leaf = name.split(".")[-1]
        kind = classify_path(name)
        blocked = False
        if freeze_vision and kind == "vision":
            blocked = True
        if freeze_router and kind == "router":
            blocked = True
        if freeze_packed_experts and kind == "expert":
            blocked = True
        if freeze_indexer and kind == "indexer":
            blocked = True
        if freeze_hyper_connections and kind == "hyper_connection":
            blocked = True
        if freeze_embeddings and kind == "embedding":
            blocked = True
        if freeze_lm_head and kind == "lm_head":
            blocked = True
        if blocked:
            exclude_paths.append(name)
            continue
        if target_mode == "explicit":
            if leaf in set(explicit_targets):
                include_paths.append(name)
            else:
                exclude_paths.append(name)
        elif target_mode == "all_linear_filtered":
            include_paths.append(name)
        else:
            # auto_stage1
            if leaf in preferred:
                include_paths.append(name)
            else:
                exclude_paths.append(name)

    # PEFT matches by leaf name. Expand to unique leaf names that survived filtering.
    leaves = sorted({p.split(".")[-1] for p in include_paths})
    if not leaves:
        notes.append(
            "No Linear LoRA targets survived filtering. This is fatal — "
            "inspect named_modules() before retrying."
        )

    # Collision: down_proj exists on shared_experts (Linear, keep) and experts (Parameter, skip).
    # PEFT wraps Linear only, so leaf name 'down_proj' is safe IF experts stay Parameters.
    notes.append(
        "Leaf-name collision: down_proj is Linear on shared_experts/dense MLP and "
        "nn.Parameter on packed experts. PEFT LoRA wraps Linear only; packed expert "
        "Parameters are not adapted."
    )
    if router_experiment_enabled:
        notes.append(
            "router_experiment.enabled=true — still not training the router unless "
            "train_router=true. Stage I default keeps the router frozen."
        )

    modules_to_save: list[str] = []
    if router_experiment_enabled and router_experiment_train_router:
        notes.append(
            "Router training requested. Glm5NextTextTopkRouter.weight is nn.Parameter, "
            "not Linear — LoRA cannot wrap it. modules_to_save would full-FT the router. "
            "Stage I refuses this unless you explicitly understand the risk."
        )
        # Still do not add it in Stage I even if the flag is on — the trainer
        # will refuse unless RAIDEN_ALLOW_ROUTER_FT=1.
        import os

        if os.environ.get("RAIDEN_ALLOW_ROUTER_FT") == "1":
            modules_to_save.append("gate")
        else:
            notes.append(
                "Set RAIDEN_ALLOW_ROUTER_FT=1 to actually full-FT the router. "
                "Default: ignored."
            )

    if not include_paths:
        raise RuntimeError(
            "LoRA target discovery produced an empty set. "
            "Run scripts/inspect_model.py --full and inspect candidate Linear names."
        )

    return LoRAPlan(
        target_modules=leaves,
        target_full_paths=sorted(include_paths),
        excluded_full_paths=sorted(exclude_paths),
        modules_to_save=modules_to_save,
        freeze_patterns=freeze_patterns,
        notes=notes,
    )


def peft_lora_config(plan: LoRAPlan, *, r: int, alpha: int, dropout: float, bias: str = "none"):
    from peft import LoraConfig, TaskType

    kwargs: dict[str, Any] = {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": bias,
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": plan.target_modules,
        "modules_to_save": list(plan.modules_to_save) or None,
    }
    # exclude_modules is available in recent PEFT; keep optional.
    exclude = []
    if any("visual" in p or p == "visual" for p in plan.freeze_patterns):
        exclude.append("visual")
    try:
        return LoraConfig(**kwargs, exclude_modules=exclude or None)
    except TypeError:
        return LoraConfig(**kwargs)


def dump_census_json(census: ModuleCensus, plan: LoRAPlan | None = None) -> str:
    payload = {
        "architecture": census.architecture,
        "n_modules": census.n_modules,
        "n_layers": census.n_layers,
        "layer_types": census.layer_types,
        "mlp_layer_types": census.mlp_layer_types,
        "param_count_by_kind": census.param_count_by_kind,
        "n_linear": len(census.linear_modules),
        "n_conv1d": len(census.conv1d_modules),
        "n_router": len(census.router_modules),
        "n_expert": len(census.expert_modules),
        "n_vision": len(census.vision_modules),
        "n_indexer": len(census.indexer_modules),
        "linear_leaf_names": sorted({n.split(".")[-1] for n in census.linear_modules}),
        "router_sample": census.router_modules[:20],
        "expert_sample": census.expert_modules[:20],
        "vision_sample": census.vision_modules[:20],
        "indexer_sample": census.indexer_modules[:20],
        "parameter_only_sample": census.parameter_only_modules[:20],
    }
    if plan:
        payload["lora_plan"] = {
            "target_modules": plan.target_modules,
            "n_target_paths": len(plan.target_full_paths),
            "n_excluded_paths": len(plan.excluded_full_paths),
            "modules_to_save": plan.modules_to_save,
            "freeze_patterns": plan.freeze_patterns,
            "notes": plan.notes,
            "target_path_sample": plan.target_full_paths[:40],
        }
    return json.dumps(payload, indent=2, ensure_ascii=False)

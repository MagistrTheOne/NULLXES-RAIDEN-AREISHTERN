"""QLoRA model load for Glm5Next. No Llama shortcuts. No silent method switch."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

from raiden.compatibility import RaidenCompatibilityError, assert_qlora_ready
from raiden.expert_nf4 import (
    count_nf4_expert_modules,
    expert_nf4_load_hooks,
    install_expert_nf4_forward,
    resolve_packed_expert_policy,
)
from raiden.freeze import apply_freeze, assert_router_frozen
from raiden.lora import census_from_model, dump_census_json, peft_lora_config, plan_stage1_lora

logger = logging.getLogger("raiden")


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _resolve_causal_class():
    """Glm5Next is a multimodal LM; prefer AutoModelForImageTextToText, then CausalLM."""
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except Exception:
        pass
    try:
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM
    except Exception:
        pass
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def build_bnb_config(qlora_cfg) -> BitsAndBytesConfig:
    if qlora_cfg.method != "qlora" or int(qlora_cfg.bits) != 4:
        raise RaidenCompatibilityError(
            f"RAIDEN Stage I is QLoRA-only. Refusing method={qlora_cfg.method!r} bits={qlora_cfg.bits}."
        )
    kwargs = dict(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=qlora_cfg.double_quant,
        bnb_4bit_quant_type=qlora_cfg.quant_type,
        bnb_4bit_compute_dtype=_dtype(qlora_cfg.compute_dtype),
    )
    # quant_storage is supported on recent bitsandbytes/transformers.
    try:
        return BitsAndBytesConfig(
            **kwargs,
            bnb_4bit_quant_storage=_dtype(qlora_cfg.quant_storage),
            llm_int8_skip_modules=list(qlora_cfg.skip_modules),
        )
    except TypeError:
        try:
            return BitsAndBytesConfig(**kwargs, llm_int8_skip_modules=list(qlora_cfg.skip_modules))
        except TypeError:
            return BitsAndBytesConfig(**kwargs)


def _assert_linear_quantized(model: Any) -> None:
    n_linear4 = 0
    n_linear = 0
    for _, mod in model.named_modules():
        cls = type(mod).__name__
        if cls in {"Linear4bit", "Params4bit"} or "4bit" in cls.lower():
            n_linear4 += 1
        elif cls == "Linear":
            n_linear += 1
    logger.info("quant census: Linear4bit-like=%s remaining nn.Linear=%s", n_linear4, n_linear)
    if n_linear4 == 0:
        raise RaidenCompatibilityError(
            "QLoRA load finished but no Linear4bit modules were found. "
            "bitsandbytes did not quantize this architecture. "
            "Refusing to continue (this would silently become LoRA/full-FT)."
        )


def load_tokenizer_and_processor(model_id: str, token: str | None = None):
    tok_kwargs = dict(trust_remote_code=True, token=token)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, **tok_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AutoProcessor failed (%s); falling back to AutoTokenizer", exc)
    tokenizer = None
    if processor is not None and hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        tokenizer = processor.tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, processor


def load_quantized_base(cfg):
    """Load GLM-5.3-Flash as bitsandbytes 4-bit. No LoRA yet. No method fallback."""
    report = assert_qlora_ready(cfg)
    for w in report.warnings:
        logger.warning("%s", w)

    token = os.environ.get("HF_TOKEN")
    bnb = build_bnb_config(cfg.qlora)
    auto_cls = _resolve_causal_class()
    expert_policy = resolve_packed_expert_policy(cfg.qlora.packed_expert_policy)
    logger.info(
        "loading %s via %s + BitsAndBytes 4-bit NF4 (packed_expert_policy=%s)",
        cfg.base_model,
        auto_cls.__name__,
        expert_policy,
    )

    # device_map="auto" sees BF16 packed experts (~610 GiB) and offloads to CPU.
    # BnB then aborts. nf4_freeze forces GPU 0; the load hook NF4s experts on assign.
    if expert_policy == "nf4_freeze":
        device_map: Any = {"": 0}
    elif cfg.train.distributed_strategy == "device_map_auto":
        device_map = "auto"
    else:
        device_map = None

    model_kwargs: dict[str, Any] = dict(
        quantization_config=bnb,
        device_map=device_map,
        trust_remote_code=True,
        token=token,
        attn_implementation=os.environ.get("RAIDEN_ATTN_IMPL", "sdpa"),
    )
    try:
        hf_cfg = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True, token=token)
        if getattr(hf_cfg, "quantization_config", None) and cfg.qlora.refuse_native_fp8:
            from raiden.compatibility import _is_native_fp8

            raw = hf_cfg.to_dict()
            if _is_native_fp8(raw) and not cfg.qlora.allow_fp8_dequant:
                raise RaidenCompatibilityError(
                    "Loaded config still contains native FP8 quantization_config. "
                    "Point base_model at zai-org/GLM-5.3-Flash-BF16."
                )
    except RaidenCompatibilityError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre-load AutoConfig probe failed: %s", exc)

    try:
        if expert_policy == "nf4_freeze":
            install_expert_nf4_forward()
            with expert_nf4_load_hooks():
                model = auto_cls.from_pretrained(cfg.base_model, **model_kwargs)
            n_nf4 = count_nf4_expert_modules(model)
            if n_nf4 == 0:
                raise RaidenCompatibilityError(
                    "packed_expert_policy=nf4_freeze but no Glm5NextTextExperts were NF4-quantized "
                    "during load. Refusing to continue with BF16 packed experts (will not fit)."
                )
            logger.info("packed experts NF4-ready modules: %s", n_nf4)
        else:
            model = auto_cls.from_pretrained(cfg.base_model, **model_kwargs)
    except RaidenCompatibilityError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RaidenCompatibilityError(
            f"Failed to load {cfg.base_model} in 4-bit QLoRA mode via {auto_cls.__name__}: {exc}. "
            "Refusing to retry as bf16 LoRA / full-FT."
        ) from exc

    _assert_linear_quantized(model)
    return model


def load_qlora_model(cfg, inspection_out: str | None = None):
    """Load GLM-5.3-Flash BF16 → bitsandbytes 4-bit → PEFT LoRA, Stage I freeze."""
    model = load_quantized_base(cfg)

    from peft import prepare_model_for_kbit_training

    gc_kwargs = cfg.train.gradient_checkpointing_kwargs or {"use_reentrant": False}
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    census = census_from_model(model)
    plan = plan_stage1_lora(
        model,
        freeze_router=cfg.freeze.freeze_router,
        freeze_vision=cfg.freeze.freeze_vision,
        freeze_packed_experts=cfg.freeze.freeze_packed_experts,
        freeze_indexer=cfg.freeze.freeze_indexer,
        freeze_hyper_connections=cfg.freeze.freeze_hyper_connections,
        freeze_embeddings=cfg.freeze.freeze_embeddings,
        freeze_lm_head=cfg.freeze.freeze_lm_head,
        router_experiment_enabled=cfg.freeze.router_experiment_enabled,
        router_experiment_train_router=cfg.freeze.router_experiment_train_router,
        target_mode=cfg.lora.target_mode,
        explicit_targets=cfg.lora.explicit_targets,
    )
    blob = dump_census_json(census, plan)
    logger.info("LoRA targets: %s", plan.target_modules)
    if inspection_out:
        Path_write = __import__("pathlib").Path
        Path_write(inspection_out).write_text(blob, encoding="utf-8")

    from peft import get_peft_model

    lora_cfg = peft_lora_config(
        plan, r=cfg.lora.r, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, bias=cfg.lora.bias
    )
    model = get_peft_model(model, lora_cfg)
    stats = apply_freeze(model, cfg.freeze)
    if cfg.freeze.freeze_router:
        assert_router_frozen(model)
    model.print_trainable_parameters()
    logger.info("freeze stats: %s", stats)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, plan, census

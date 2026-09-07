"""Hard compatibility gates for GLM-5.3-Flash QLoRA.

This module NEVER silently switches training method.
If QLoRA cannot be executed correctly, it raises RaidenCompatibilityError.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from raiden.family import FORBIDDEN_BASE_REASON, forbidden_base_hit
from raiden.identity import BASE_ARCHITECTURE, BASE_MODEL_TYPE, BASE_REPO_BF16, BASE_REPO_PUBLIC


class RaidenCompatibilityError(RuntimeError):
    """Raised when QLoRA cannot start without substituting another method."""


MIN_TRANSFORMERS = (5, 16, 0)
MIN_PEFT = (0, 17, 0)
MIN_BNB = (0, 45, 0)
MIN_TRL = (0, 21, 0)
MIN_ACCELERATE = (1, 2, 0)

NATIVE_FP8_MARKERS = ("compressed-tensors", "e4m3", "float8", "fp8")


def _ver_tuple(mod) -> tuple[int, ...]:
    raw = getattr(mod, "__version__", "0.0.0").split("+")[0]
    parts = []
    for p in raw.split("."):
        n = ""
        for ch in p:
            if ch.isdigit():
                n += ch
            else:
                break
        parts.append(int(n) if n else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RaidenCompatibilityError(msg)


@dataclass
class CompatReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.ok:
            body = "\n".join(f"  - {e}" for e in self.errors)
            raise RaidenCompatibilityError(
                "RAIDEN_QLORA_INCOMPATIBLE: refusing to start. "
                "QLoRA was requested and will not be silently replaced by LoRA-bf16 "
                "or full fine-tuning.\n"
                f"{body}"
            )


def parse_hf_config(model_id_or_path: str, token: str | None = None) -> dict:
    path = Path(model_id_or_path)
    cfg_path = path / "config.json" if path.exists() else None
    if cfg_path and cfg_path.is_file():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=model_id_or_path,
            filename="config.json",
            token=token or os.environ.get("HF_TOKEN"),
        )
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RaidenCompatibilityError(
            f"Cannot read config.json for {model_id_or_path!r}: {exc}"
        ) from exc


def inspect_packages() -> dict[str, Any]:
    facts: dict[str, Any] = {"python": sys.version}
    for name in (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "trl",
        "datasets",
        "unsloth",
    ):
        try:
            mod = importlib.import_module(name)
            facts[name] = {
                "present": True,
                "version": getattr(mod, "__version__", "unknown"),
                "file": getattr(mod, "__file__", None),
            }
        except Exception as exc:  # noqa: BLE001
            facts[name] = {"present": False, "error": str(exc)}
    return facts


def _probe_bitsandbytes_cuda() -> tuple[bool, str]:
    """bitsandbytes>=0.44 removed COMPILED_WITH_CUDA. Probe the loaded native lib."""
    try:
        import bitsandbytes as bnb
        from bitsandbytes import cextension
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc}"

    backend = getattr(cextension, "BNB_BACKEND", None)
    lib = getattr(cextension, "lib", None)
    compiled = getattr(lib, "compiled_with_cuda", None)
    lib_name = type(lib).__name__ if lib is not None else None
    if lib_name == "ErrorHandlerMockBNBNativeLibrary":
        return False, f"native CUDA lib missing backend={backend!r} lib={lib_name}"

    cuda_ok = compiled is True or (str(backend).upper() == "CUDA" and compiled is not False)
    if not cuda_ok:
        return False, f"backend={backend!r} compiled_with_cuda={compiled} lib={lib_name}"

    try:
        import torch

        if torch.cuda.is_available():
            layer = bnb.nn.Linear4bit(32, 32, bias=False, quant_type="nf4")
            layer = layer.to(device="cuda")
            x = torch.randn(2, 32, device="cuda", dtype=torch.float16)
            _ = layer(x)
            del layer, x
            torch.cuda.synchronize()
            return True, f"backend={backend} compiled_with_cuda={compiled} linear4bit_ok=true"
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"backend={backend} compiled_with_cuda={compiled} linear4bit_failed={exc}"
        )
    return True, f"backend={backend} compiled_with_cuda={compiled} linear4bit_skipped=no_cuda"


def check_glm5_next_in_transformers() -> tuple[bool, str]:
    try:
        from transformers import AutoConfig

        mapping = getattr(AutoConfig, "_model_mapping", None) or {}
        names = set()
        for key in mapping:
            names.add(str(key))
        try:
            from transformers.models.glm5_next.configuration_glm5_next import (  # type: ignore
                Glm5NextConfig,
            )

            return True, Glm5NextConfig.model_type
        except Exception:
            pass
        try:
            import transformers.models.glm5_next as m  # noqa: F401

            return True, "glm5_next"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _is_native_fp8(cfg: dict) -> bool:
    q = cfg.get("quantization_config") or {}
    blob = json.dumps(q).lower()
    if any(m in blob for m in NATIVE_FP8_MARKERS):
        return True
    fmt = str(q.get("fmt", "")).lower()
    return fmt in {"e4m3", "e5m2", "float8"}


def check_environment(
    *,
    base_model: str,
    backend: str = "hf_qlora",
    refuse_native_fp8: bool = True,
    allow_fp8_dequant: bool = False,
    require_expert_4bit: bool = False,
    require_linear_4bit: bool = True,
    token: str | None = None,
    load_weights: bool = False,
) -> CompatReport:
    report = CompatReport(ok=True)
    banned = forbidden_base_hit(base_model)
    if banned:
        report.errors.append(
            f"base_model={base_model!r} is forbidden ({banned}). {FORBIDDEN_BASE_REASON}"
        )
        report.ok = False
        return report

    facts = inspect_packages()
    report.facts["packages"] = facts

    def need(pkg: str, min_v: tuple[int, int, int]) -> None:
        info = facts.get(pkg) or {}
        if not info.get("present"):
            report.errors.append(
                f"{pkg} is not installed. QLoRA backend requires it. "
                f"Install the version in requirements.txt. No fallback method will be used."
            )
            return
        mod = importlib.import_module(pkg)
        ver = _ver_tuple(mod)
        if ver < min_v:
            report.errors.append(
                f"{pkg}=={info.get('version')} is too old (need >={'.'.join(map(str, min_v))}). "
                f"GLM-5.3-Flash (Glm5Next) is not in older Transformers/PEFT stacks."
            )

    need("transformers", MIN_TRANSFORMERS)
    need("peft", MIN_PEFT)
    need("bitsandbytes", MIN_BNB)
    need("trl", MIN_TRL)
    need("accelerate", MIN_ACCELERATE)
    need("torch", (2, 4, 0))

    ok_arch, arch_msg = check_glm5_next_in_transformers()
    report.facts["transformers_glm5_next"] = {"ok": ok_arch, "detail": arch_msg}
    if not ok_arch:
        report.errors.append(
            "Installed transformers does not expose glm5_next / Glm5NextForConditionalGeneration. "
            "Upgrade to transformers>=5.16.0 (the version that ships GLM-5.3-Flash). "
            "Refusing to train under a wrong architecture mapping."
        )

    bnb_ok, bnb_fact = _probe_bitsandbytes_cuda()
    report.facts["bitsandbytes_cuda"] = bnb_fact
    if not bnb_ok:
        report.errors.append(
            "bitsandbytes CUDA/4-bit probe failed. QLoRA cannot run. "
            "Refusing CPU/LoRA-bf16 fallback. "
            f"Detail: {bnb_fact}"
        )

    try:
        import torch

        report.facts["cuda_available"] = bool(torch.cuda.is_available())
        report.facts["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if not torch.cuda.is_available():
            report.errors.append(
                "torch.cuda.is_available() is False. QLoRA on RunPod requires CUDA. Refusing."
            )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"torch CUDA probe failed: {exc}")

    cfg = parse_hf_config(base_model, token=token)
    arch = (cfg.get("architectures") or [None])[0]
    model_type = cfg.get("model_type")
    text_cfg = cfg.get("text_config") or {}
    vision_cfg = cfg.get("vision_config") or {}
    report.facts["hf_config"] = {
        "architectures": cfg.get("architectures"),
        "model_type": model_type,
        "num_hidden_layers": text_cfg.get("num_hidden_layers"),
        "n_routed_experts": text_cfg.get("n_routed_experts"),
        "n_shared_experts": text_cfg.get("n_shared_experts"),
        "num_experts_per_tok": text_cfg.get("num_experts_per_tok"),
        "has_vision_config": bool(vision_cfg),
        "native_fp8": _is_native_fp8(cfg),
        "quantization_config_present": bool(cfg.get("quantization_config")),
        "layer_types_preview": (text_cfg.get("layer_types") or [])[:8],
        "mlp_layer_types_preview": (text_cfg.get("mlp_layer_types") or [])[:8],
        "transformers_version_in_card": cfg.get("transformers_version"),
    }

    if arch != BASE_ARCHITECTURE or model_type not in {BASE_MODEL_TYPE, "glm5_next"}:
        report.errors.append(
            f"Expected {BASE_ARCHITECTURE} / model_type={BASE_MODEL_TYPE}, "
            f"got architectures={cfg.get('architectures')!r} model_type={model_type!r}. "
            "Refusing to apply Llama-style assumptions."
        )

    native_fp8 = _is_native_fp8(cfg)
    if native_fp8 and refuse_native_fp8 and not allow_fp8_dequant:
        report.errors.append(
            f"Checkpoint {base_model!r} is a native FP8 (compressed-tensors/e4m3) weight dump. "
            "bitsandbytes QLoRA needs unquantized Linear weights. "
            f"Use the BF16 training weights: {BASE_REPO_BF16} "
            f"(product base remains {BASE_REPO_PUBLIC}). "
            "Set qlora.allow_fp8_dequant=true only if you have a verified dequant path. "
            "Refusing to silently 4-bit-quantize an already-FP8 MoE checkpoint."
        )
    elif native_fp8 and allow_fp8_dequant:
        report.warnings.append(
            "allow_fp8_dequant=true: will attempt to dequantize native FP8 then apply bitsandbytes 4-bit. "
            "This path is experimental. Prefer GLM-5.3-Flash-BF16."
        )

    # Packed experts are nn.Parameter, not nn.Linear. Document, do not hide.
    report.facts["packed_experts"] = {
        "class": "Glm5NextTextExperts",
        "storage": "nn.Parameter (gate_up_proj, down_proj), NOT nn.Linear",
        "bnb_linear4bit_applies": False,
        "peft_lora_on_parameter": False,
        "stage1_policy": "QLoRA nn.Linear + frozen packed experts (NF4 on single-B300)",
    }
    report.warnings.append(
        "GLM-5.3-Flash MoE experts are packed nn.Parameter tensors. "
        "bitsandbytes Linear4bit and PEFT LoRA target nn.Linear only. "
        "Stage I QLoRA quantizes + adapts Linear modules "
        "(attention, dense MLP, shared expert) and FREEZES packed experts. "
        "On a single B300, experts are stored NF4 (packed_expert_policy=nf4_freeze) "
        "because BF16 packed experts (~610 GiB) do not fit 275 GiB. "
        "This is still QLoRA, not a method switch."
    )
    if require_expert_4bit:
        report.errors.append(
            "qlora.require_expert_4bit=true means BnB Linear4bit on packed experts, "
            "which cannot exist (they are nn.Parameter). "
            "Use packed_expert_policy=nf4_freeze instead: frozen NF4 Parameters, "
            "LoRA still Linear-only. Set require_expert_4bit=false."
        )

    if require_linear_4bit:
        report.facts["require_linear_4bit"] = True

    if backend == "unsloth":
        u = facts.get("unsloth") or {}
        if not u.get("present"):
            report.errors.append(
                "qlora.backend=unsloth but unsloth is not installed. "
                "Refusing to fall back to hf_qlora silently. Install unsloth or set backend=hf_qlora."
            )
        else:
            report.errors.append(
                "qlora.backend=unsloth is disabled for RAIDEN Stage I. "
                "Unsloth's MoE guide states 4-bit QLoRA is not supported because "
                "bitsandbytes cannot quantize packed experts, and Unsloth would "
                "steer toward bf16 LoRA. RAIDEN will not silently change method. "
                "Use backend=hf_qlora (Transformers + PEFT + bitsandbytes Linear QLoRA)."
            )
    elif backend != "hf_qlora":
        report.errors.append(
            f"Unknown qlora.backend={backend!r}. Supported: hf_qlora. "
            "unsloth is explicitly rejected for Stage I QLoRA (see above)."
        )

    if load_weights:
        report.warnings.append("load_weights=true is reserved for inspect_model --full")

    report.ok = not report.errors
    return report


def assert_qlora_ready(cfg) -> CompatReport:
    report = check_environment(
        base_model=cfg.base_model,
        backend=cfg.qlora.backend,
        refuse_native_fp8=cfg.qlora.refuse_native_fp8,
        allow_fp8_dequant=cfg.qlora.allow_fp8_dequant,
        require_expert_4bit=cfg.qlora.require_expert_4bit,
        require_linear_4bit=cfg.qlora.require_linear_4bit,
    )
    report.raise_if_failed()
    return report

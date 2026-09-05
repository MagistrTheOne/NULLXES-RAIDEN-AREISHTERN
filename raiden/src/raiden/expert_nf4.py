"""NF4 storage for Glm5Next packed expert Parameters.

BnB Linear4bit cannot wrap `gate_up_proj` / `down_proj` (they are 3D nn.Parameter).
Leaving them BF16 needs ~610 GiB. One B300 is 275 GiB, so freeze_bf16 cannot load.

This is still Stage I QLoRA: LoRA on nn.Linear, experts frozen. Not LoRA-bf16.
Forward dequantizes only the experts that actually fire (same loop as upstream).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("raiden")

EXPERT_PARAM_NAMES = frozenset({"gate_up_proj", "down_proj"})
EXPERT_CLASS_NAME = "Glm5NextTextExperts"
VRAM_BF16_EXPERTS_MIN_BYTES = 400 * 1024**3  # below this, freeze_bf16 cannot fit


def gpu_vram_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return None


def resolve_packed_expert_policy(policy: str) -> str:
    policy = (policy or "nf4_freeze").strip()
    if policy != "freeze_bf16":
        return policy
    vram = gpu_vram_bytes()
    if vram is None:
        return policy
    if vram < VRAM_BF16_EXPERTS_MIN_BYTES:
        logger.warning(
            "packed_expert_policy=freeze_bf16 needs ~610 GiB resident experts; "
            "this GPU has %.1f GiB. Switching to nf4_freeze (frozen NF4 Parameters). "
            "LoRA targets stay Linear-only.",
            vram / 1024**3,
        )
        return "nf4_freeze"
    return policy


def _module_is_packed_experts(mod: Any) -> bool:
    return type(mod).__name__ == EXPERT_CLASS_NAME


def _quantize_matrix(weight_2d):
    import bitsandbytes.functional as Fbnb
    import torch

    w = weight_2d.detach().contiguous()
    if w.device.type != "cuda":
        w = w.to(device="cuda", dtype=torch.bfloat16)
    elif w.dtype != torch.bfloat16:
        w = w.to(dtype=torch.bfloat16)
    q, qs = Fbnb.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
    return q, qs


def _dequantize_matrix(q, qs, dtype):
    import bitsandbytes.functional as Fbnb

    return Fbnb.dequantize_4bit(q, qs).to(dtype=dtype)


def _absmax_float(qs):
    import bitsandbytes.functional as Fbnb

    if getattr(qs, "nested", False) and qs.state2 is not None:
        absmax = Fbnb.dequantize_blockwise(qs.absmax, qs.state2)
        if qs.offset is not None:
            absmax = absmax + qs.offset
        return absmax.float()
    return qs.absmax.float()


def _split_packed_nf4(q, qs, n_exp: int, out: int, inn: int):
    """Turn one NF4 packed [E, out*in] quant into per-expert (q, QuantState) views."""
    import torch
    from bitsandbytes.functional import QuantState

    n_el = out * inn
    blocksize = int(qs.blocksize)
    if n_el % blocksize != 0:
        raise ValueError(f"expert matrix {out}x{inn} is not divisible by NF4 blocksize {blocksize}")
    blocks = n_el // blocksize
    q_per = n_el // 2
    q_flat = q.reshape(-1)
    absmax = _absmax_float(qs).reshape(-1)
    if int(q_flat.numel()) != n_exp * q_per or int(absmax.numel()) != n_exp * blocks:
        raise ValueError(
            f"NF4 layout mismatch q={int(q_flat.numel())} absmax={int(absmax.numel())} "
            f"expected q={n_exp * q_per} absmax={n_exp * blocks}"
        )
    rows = []
    for i in range(n_exp):
        qs_i = QuantState(
            absmax=absmax[i * blocks : (i + 1) * blocks],
            shape=torch.Size((out, inn)),
            code=qs.code,
            blocksize=blocksize,
            quant_type=qs.quant_type,
            dtype=qs.dtype,
        )
        rows.append((q_flat[i * q_per : (i + 1) * q_per], qs_i))
    return rows


def attach_packed_expert_nf4(module: Any, param_name: str, value) -> int:
    """Quantize a packed [E, out, in] tensor into per-expert NF4 rows. Returns E."""
    import bitsandbytes.functional as Fbnb
    import torch

    if param_name not in EXPERT_PARAM_NAMES:
        raise ValueError(param_name)
    if value is None:
        raise ValueError("expert tensor is None")
    w = value.detach()
    if w.ndim != 3:
        raise ValueError(f"expected packed 3D expert tensor, got {tuple(w.shape)}")
    n_exp, out, inn = (int(x) for x in w.shape)
    w2 = w.to(device="cuda", dtype=torch.bfloat16).contiguous().reshape(n_exp, out * inn)
    try:
        q, qs = Fbnb.quantize_4bit(w2, quant_type="nf4", compress_statistics=True)
        rows = _split_packed_nf4(q, qs, n_exp, out, inn)
        how = "batched"
    except Exception as exc:  # noqa: BLE001
        logger.warning("batched NF4 split failed (%s); per-expert fallback", exc)
        rows = [_quantize_matrix(w2[i].reshape(out, inn)) for i in range(n_exp)]
        how = "per-expert"
    del w, w2
    store = getattr(module, "_raiden_nf4", None)
    if store is None:
        store = {}
        module._raiden_nf4 = store
    store[param_name] = rows
    if not hasattr(module, "_raiden_nf4_shape"):
        module._raiden_nf4_shape = {}
    module._raiden_nf4_shape[param_name] = (n_exp, out, inn)
    logger.info(
        "NF4 packed expert %s experts=%s shape=%s via=%s",
        param_name,
        n_exp,
        (n_exp, out, inn),
        how,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n_exp


def _nf4_forward(self, hidden_states, top_k_index, top_k_weights):
    import torch
    import torch.nn.functional as F

    store = getattr(self, "_raiden_nf4", None)
    if not store or "gate_up_proj" not in store or "down_proj" not in store:
        return self._raiden_orig_expert_forward(hidden_states, top_k_index, top_k_weights)
    final = torch.zeros_like(hidden_states)
    dtype = hidden_states.dtype
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            i = int(expert_idx.item())
            w_gu = _dequantize_matrix(*store["gate_up_proj"][i], dtype)
            w_dn = _dequantize_matrix(*store["down_proj"][i], dtype)
            current = self._apply_gate(F.linear(hidden_states[token_idx], w_gu))
            current = F.linear(current, w_dn) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
    return final


def install_expert_nf4_forward() -> None:
    try:
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot patch Glm5NextTextExperts.forward: {exc}") from exc
    if getattr(Glm5NextTextExperts.forward, "_raiden_nf4", False):
        return
    Glm5NextTextExperts._raiden_orig_expert_forward = Glm5NextTextExperts.forward
    _nf4_forward._raiden_nf4 = True
    Glm5NextTextExperts.forward = _nf4_forward
    logger.info("patched Glm5NextTextExperts.forward for per-expert NF4 dequant")


def count_nf4_expert_modules(model: Any) -> int:
    n = 0
    for _, mod in model.named_modules():
        store = getattr(mod, "_raiden_nf4", None)
        if store and "gate_up_proj" in store and "down_proj" in store:
            n += 1
    return n


def _dummy_param(like):
    import torch

    device = getattr(like, "device", None)
    if device is None or (hasattr(device, "type") and device.type == "meta"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)
    p._is_hf_initialized = True
    return p


@contextmanager
def expert_nf4_load_hooks():
    """Intercept Transformers v5 / Accelerate weight assignment for packed experts."""
    patches: list[tuple[Any, str, Any]] = []

    def _track(mod, name: str, wrapper) -> None:
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "_raiden_nf4_hook", False):
            return
        wrapper._raiden_nf4_hook = True
        setattr(mod, name, wrapper)
        patches.append((mod, name, orig))

    try:
        import transformers.core_model_loading as cml

        orig_set = cml.set_param_for_module

        def set_param_for_module(model, target_name, param_value, loading_info, hf_quantizer):
            module_path, _, param_name = target_name.rpartition(".")
            if param_name in EXPERT_PARAM_NAMES:
                module_obj = model.get_submodule(module_path) if module_path else model
                if _module_is_packed_experts(module_obj) and param_value is not None:
                    attach_packed_expert_nf4(module_obj, param_name, param_value)
                    return orig_set(model, target_name, _dummy_param(param_value), loading_info, hf_quantizer)
            return orig_set(model, target_name, param_value, loading_info, hf_quantizer)

        _track(cml, "set_param_for_module", set_param_for_module)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not hook transformers.core_model_loading.set_param_for_module: %s", exc)

    try:
        import accelerate.utils.modeling as acc_m

        orig_acc = acc_m.set_module_tensor_to_device

        def set_module_tensor_to_device(module, tensor_name, device, value=None, **kwargs):
            leaf = tensor_name.split(".")[-1]
            if (
                "." not in tensor_name
                and _module_is_packed_experts(module)
                and leaf in EXPERT_PARAM_NAMES
                and value is not None
            ):
                attach_packed_expert_nf4(module, leaf, value)
                return orig_acc(module, tensor_name, device, value=_dummy_param(value), **kwargs)
            return orig_acc(module, tensor_name, device, value=value, **kwargs)

        _track(acc_m, "set_module_tensor_to_device", set_module_tensor_to_device)
        try:
            import accelerate.utils as acc_u

            if getattr(acc_u, "set_module_tensor_to_device", None) is orig_acc:
                _track(acc_u, "set_module_tensor_to_device", set_module_tensor_to_device)
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not hook accelerate set_module_tensor_to_device: %s", exc)

    try:
        yield
    finally:
        for mod, name, orig in patches:
            setattr(mod, name, orig)

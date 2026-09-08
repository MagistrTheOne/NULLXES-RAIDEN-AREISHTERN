"""NF4 storage for Glm5Next packed expert Parameters.

BnB Linear4bit cannot wrap `gate_up_proj` / `down_proj` (they are 3D nn.Parameter).
Leaving them BF16 needs ~610 GiB. One B300 is 275 GiB, so freeze_bf16 cannot load.

This is still Stage I QLoRA: LoRA on nn.Linear, experts frozen. Not LoRA-bf16.
Forward dequantizes only the experts that actually fire (same loop as upstream).
Frozen W still needs dL/dx = Wᵀ · dL/dy. Dequant is detached; Linear/gate stay in the graph.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from raiden.expert_nf4_cache import (
    apply_rows_to_module,
    checkpoint_key_to_packed_key,
    expert_nf4_cache_dir,
    load_expert_blob,
    meta_shape_for_skipped_read,
    resolve_cached_key,
    save_expert_blob,
)

logger = logging.getLogger("raiden")

EXPERT_PARAM_NAMES = frozenset({"gate_up_proj", "down_proj"})
EXPERT_CLASS_NAME = "Glm5NextTextExperts"
VRAM_BF16_EXPERTS_MIN_BYTES = 400 * 1024**3  # below this, freeze_bf16 cannot fit
# bitsandbytes CUDA kernels take int32 element counts. 288*4096*4096 = 4.83e9 > 2^31-1.
_NF4_MAX_ELEMENTS = 2_000_000_000


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


def nf4_expert_chunk_size(n_exp: int, out: int, inn: int) -> int:
    """Largest expert batch whose BF16 numel fits in a signed int32 CUDA launch."""
    per = max(1, out * inn)
    return max(1, min(n_exp, _NF4_MAX_ELEMENTS // per))


def quantize_packed_expert_tensor(value, log_name: str = "") -> tuple[list, tuple[int, int, int]]:
    """NF4 a packed [E, out, in] tensor. Uploads int32-safe chunks, never the full 4.8e9 elems."""
    import bitsandbytes.functional as Fbnb
    import torch

    if value is None:
        raise ValueError("expert tensor is None")
    w = value.detach()
    if w.ndim != 3:
        raise ValueError(f"expected packed 3D expert tensor, got {tuple(w.shape)}")
    n_exp, out, inn = (int(x) for x in w.shape)
    chunk = nf4_expert_chunk_size(n_exp, out, inn)
    rows: list = []
    n_chunks = (n_exp + chunk - 1) // chunk
    for i, start in enumerate(range(0, n_exp, chunk)):
        sl = w[start : start + chunk]
        n = int(sl.shape[0])
        sl = sl.to(device="cuda", dtype=torch.bfloat16).contiguous()
        q, qs = Fbnb.quantize_4bit(
            sl.reshape(n * out, inn).contiguous(),
            quant_type="nf4",
            compress_statistics=True,
        )
        rows.extend(_split_packed_nf4(q, qs, n, out, inn))
        del q, qs, sl
        logger.info(
            "NF4 %s chunk %s/%s experts %s-%s/%s via=chunk%s",
            log_name or "packed",
            i + 1,
            n_chunks,
            start,
            start + n,
            n_exp,
            chunk,
        )
    del w
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, (n_exp, out, inn)


def attach_packed_expert_nf4(
    module: Any,
    param_name: str,
    value,
    weight_name: str | None = None,
) -> int:
    """Attach per-expert NF4 rows from cache or by quantizing `value`. Returns E."""
    import torch

    if param_name not in EXPERT_PARAM_NAMES:
        raise ValueError(param_name)
    cache_dir = expert_nf4_cache_dir()
    key = weight_name or param_name
    cached = resolve_cached_key(cache_dir, weight_name) if cache_dir is not None and weight_name else None
    if cached:
        rows, shape = load_expert_blob(cache_dir, cached, device="cuda")
        apply_rows_to_module(module, param_name, rows, shape)
        logger.info(
            "NF4 packed expert %s experts=%s shape=%s via=cache",
            key,
            shape[0],
            shape,
        )
        return int(shape[0])
    if value is None:
        raise ValueError(f"expert tensor is None and cache miss for {key}")
    if hasattr(value, "is_meta") and value.is_meta:
        raise ValueError(f"expert tensor is meta and cache miss for {key}")
    rows, shape = quantize_packed_expert_tensor(value, log_name=key)
    apply_rows_to_module(module, param_name, rows, shape)
    how = f"chunk{nf4_expert_chunk_size(*shape)}"
    if cache_dir is not None and weight_name:
        save_expert_blob(cache_dir, weight_name, rows, shape)
        how = f"{how}+write"
    logger.info(
        "NF4 packed expert %s experts=%s shape=%s via=%s",
        key,
        shape[0],
        shape,
        how,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return int(shape[0])


def apply_frozen_expert(
    hidden_states,
    token_idx,
    top_k_pos,
    top_k_weights,
    w_gu,
    w_dn,
    apply_gate,
    acc,
):
    """Expert matmuls with constant W. Gradients flow to activations and router weights.

    For y = Wx with frozen W, dL/dx = Wᵀ · dL/dy must remain in the graph.
    Only the weight tensors are detached. Do not wrap this in torch.no_grad().
    """
    import torch.nn.functional as F

    current = apply_gate(F.linear(hidden_states[token_idx], w_gu.detach()))
    current = F.linear(current, w_dn.detach()) * top_k_weights[token_idx, top_k_pos, None]
    return acc.index_add(0, token_idx, current.to(dtype=acc.dtype))


def _nf4_forward(self, hidden_states, top_k_index, top_k_weights):
    import torch
    import torch.nn.functional as F

    store = getattr(self, "_raiden_nf4", None)
    if not store or "gate_up_proj" not in store or "down_proj" not in store:
        return self._raiden_orig_expert_forward(hidden_states, top_k_index, top_k_weights)
    acc = hidden_states.new_zeros(hidden_states.shape)
    dtype = hidden_states.dtype
    mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        i = int(expert_idx.item())
        with torch.no_grad():
            w_gu = _dequantize_matrix(*store["gate_up_proj"][i], dtype)
            w_dn = _dequantize_matrix(*store["down_proj"][i], dtype)
        acc = apply_frozen_expert(
            hidden_states,
            token_idx,
            top_k_pos,
            top_k_weights,
            w_gu,
            w_dn,
            self._apply_gate,
            acc,
        )
    return acc


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

    device = getattr(like, "device", None) if like is not None else None
    if device is None or (hasattr(device, "type") and device.type == "meta"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)
    p._is_hf_initialized = True
    return p


def attach_cached_experts_on_model(model: Any, cache_dir=None) -> int:
    """Fill any packed-expert module still missing NF4 rows from the disk cache."""
    cache_dir = expert_nf4_cache_dir(cache_dir)
    if cache_dir is None:
        return 0
    n = 0
    for name, mod in model.named_modules():
        if not _module_is_packed_experts(mod):
            continue
        store = getattr(mod, "_raiden_nf4", None) or {}
        for pname in EXPERT_PARAM_NAMES:
            if pname in store:
                continue
            key = f"{name}.{pname}"
            cached = resolve_cached_key(cache_dir, key)
            if not cached:
                continue
            rows, shape = load_expert_blob(cache_dir, cached, device="cuda")
            apply_rows_to_module(mod, pname, rows, shape)
            n += 1
            logger.info("NF4 packed expert %s experts=%s shape=%s via=cache-post", key, shape[0], shape)
    return n


@contextmanager
def skip_cached_expert_reads(cache_dir=None):
    """Stop Hugging Face from reading BF16 packed experts that are already NF4 on disk.

    Returns a meta tensor of the cached shape so assignment hooks still fire.
    """
    cache_dir = expert_nf4_cache_dir(cache_dir)
    if cache_dir is None:
        yield
        return

    import torch

    patches: list[tuple[Any, str, Any]] = []
    skipped_packed: set[str] = set()

    class _Handle:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __enter__(self):
            inner = self._inner
            entered = inner.__enter__() if hasattr(inner, "__enter__") else inner
            object.__setattr__(self, "_inner", entered)
            return self

        def __exit__(self, *exc):
            inner = self._inner
            if hasattr(inner, "__exit__"):
                return inner.__exit__(*exc)
            return None

        def get_tensor(self, name):
            packed = checkpoint_key_to_packed_key(name)
            cached = resolve_cached_key(cache_dir, packed) if packed else None
            if cached:
                shape = meta_shape_for_skipped_read(cache_dir, name, cached)
                if cached not in skipped_packed:
                    skipped_packed.add(cached)
                    logger.info(
                        "skip BF16 expert reads for %s (hit %s, meta %s)",
                        cached,
                        name,
                        shape,
                    )
                return torch.empty(shape, dtype=torch.bfloat16, device="meta")
            return self._inner.get_tensor(name)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _wrap(orig):
        def safe_open(*args, **kwargs):
            return _Handle(orig(*args, **kwargs))

        return safe_open

    import importlib

    candidates = [
        "safetensors",
        "safetensors.torch",
        "transformers.core_model_loading",
        "transformers.modeling_utils",
        "transformers.integrations.hub_kernels",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        orig = getattr(mod, "safe_open", None)
        if orig is None or getattr(orig, "_raiden_nf4_skip", False):
            continue
        wrapped = _wrap(orig)
        wrapped._raiden_nf4_skip = True
        setattr(mod, "safe_open", wrapped)
        patches.append((mod, "safe_open", orig))
    if not patches:
        logger.warning("could not wrap safetensors.safe_open; cached experts will still be read as BF16")
    try:
        yield
    finally:
        for mod, name, orig in patches:
            setattr(mod, name, orig)


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
                cache_dir = expert_nf4_cache_dir()
                cached = resolve_cached_key(cache_dir, target_name) if cache_dir else None
                if _module_is_packed_experts(module_obj) and (param_value is not None or cached):
                    attach_packed_expert_nf4(module_obj, param_name, param_value, weight_name=target_name)
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
                attach_packed_expert_nf4(module, leaf, value, weight_name=None)
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

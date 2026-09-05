from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.expert_nf4 import (
    VRAM_BF16_EXPERTS_MIN_BYTES,
    apply_frozen_expert,
    count_nf4_expert_modules,
    nf4_expert_chunk_size,
    resolve_packed_expert_policy,
)
from raiden.expert_nf4_cache import (
    cache_file_for_key,
    is_packed_expert_weight_key,
    resolve_cached_key,
    save_manifest,
)
from raiden.config import RaidenConfig


def test_yaml_default_is_nf4_freeze():
    src = Path(__file__).resolve().parents[1] / "configs" / "raiden_qlora.yaml"
    cfg = RaidenConfig.from_yaml(src)
    assert cfg.qlora.packed_expert_policy == "nf4_freeze"


def test_resolve_keeps_nf4():
    assert resolve_packed_expert_policy("nf4_freeze") == "nf4_freeze"


def test_resolve_upgrades_freeze_bf16_on_small_gpu(monkeypatch):
    import raiden.expert_nf4 as m

    monkeypatch.setattr(m, "gpu_vram_bytes", lambda: 275 * 1024**3)
    assert resolve_packed_expert_policy("freeze_bf16") == "nf4_freeze"
    monkeypatch.setattr(m, "gpu_vram_bytes", lambda: 800 * 1024**3)
    assert resolve_packed_expert_policy("freeze_bf16") == "freeze_bf16"
    assert VRAM_BF16_EXPERTS_MIN_BYTES > 275 * 1024**3


def test_nf4_chunk_stays_under_int32():
    gate = nf4_expert_chunk_size(288, 4096, 4096)
    down = nf4_expert_chunk_size(288, 4096, 2048)
    assert 1 <= gate <= 288
    assert 1 <= down <= 288
    assert gate * 4096 * 4096 < 2**31
    assert down * 4096 * 2048 < 2**31


def test_split_packed_nf4_layout_sizes():
    n_exp, out, inn, blocksize = 4, 64, 128, 64
    n_el = out * inn
    assert n_el % blocksize == 0
    assert n_el // 2 * n_exp == n_exp * n_el // 2
    assert n_el // blocksize * n_exp == n_exp * (out * inn // blocksize)


def test_packed_expert_key_skips_shared():
    assert is_packed_expert_weight_key(
        "model.language_model.layers.3.mlp.experts.gate_up_proj"
    )
    assert is_packed_expert_weight_key(
        "model.language_model.layers.3.mlp.experts.down_proj"
    )
    assert not is_packed_expert_weight_key(
        "model.language_model.layers.3.mlp.shared_experts.gate_up_proj"
    )
    assert not is_packed_expert_weight_key("model.language_model.layers.3.mlp.gate_proj")


def test_resolve_cached_key_prefix(tmp_path):
    key = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    cache_file_for_key(tmp_path, key).write_bytes(b"x")
    save_manifest(
        tmp_path,
        {"v": 1, "tensors": {key: {"shape": [288, 4096, 4096], "file": f"{key}.pt"}}},
    )
    assert resolve_cached_key(tmp_path, key) == key
    assert resolve_cached_key(tmp_path, key.removeprefix("model.")) == key
    assert resolve_cached_key(tmp_path, "nope.experts.gate_up_proj") is None


def _swiglu(t):
    import torch.nn.functional as F

    a, b = t.chunk(2, dim=-1)
    return F.silu(a) * b


def test_frozen_expert_backs_into_activations():
    import torch

    torch.manual_seed(0)
    x = torch.randn(5, 8, requires_grad=True)
    w_gu = torch.randn(16, 8, requires_grad=True)
    w_dn = torch.randn(8, 8, requires_grad=True)
    token_idx = torch.tensor([0, 2, 4])
    top_k_pos = torch.zeros(3, dtype=torch.long)
    router_w = torch.ones(5, 1)
    acc = torch.zeros_like(x)
    y = apply_frozen_expert(x, token_idx, top_k_pos, router_w, w_gu, w_dn, _swiglu, acc)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad[0].abs().sum() > 0
    assert x.grad[2].abs().sum() > 0
    assert w_gu.grad is None
    assert w_dn.grad is None


def test_no_grad_around_expert_math_kills_input_grad():
    import torch

    torch.manual_seed(0)
    x = torch.randn(5, 8, requires_grad=True)
    w_gu = torch.randn(16, 8)
    w_dn = torch.randn(8, 8)
    token_idx = torch.tensor([0, 2, 4])
    top_k_pos = torch.zeros(3, dtype=torch.long)
    router_w = torch.ones(5, 1)
    acc = torch.zeros_like(x)
    with torch.no_grad():
        y = apply_frozen_expert(x, token_idx, top_k_pos, router_w, w_gu, w_dn, _swiglu, acc)
    assert y.requires_grad is False


def test_count_nf4_expert_modules():
    ready = SimpleNamespace(
        _raiden_nf4={"gate_up_proj": [None], "down_proj": [None]},
    )
    half = SimpleNamespace(_raiden_nf4={"gate_up_proj": [None]})
    empty = SimpleNamespace()

    class M:
        def named_modules(self):
            return [("a", ready), ("b", half), ("c", empty)]

    assert count_nf4_expert_modules(M()) == 1

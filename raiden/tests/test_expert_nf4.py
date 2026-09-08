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
    estimated_nf4_cache_bytes,
    expert_cache_preflight,
    inspect_cache_state,
    is_packed_expert_weight_key,
    manifest_fingerprint,
    refuse_unready_expert_cache,
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


def test_routed_expert_linear_keys_match_official_bf16():
    from raiden.expert_nf4_cache import expert_storage_layout, is_routed_expert_linear_key

    k = "model.language_model.layers.10.mlp.experts.0.gate_proj.weight"
    assert is_routed_expert_linear_key(k)
    assert is_routed_expert_linear_key(
        "model.language_model.layers.10.mlp.experts.287.down_proj.weight"
    )
    assert not is_routed_expert_linear_key(
        "model.language_model.layers.10.mlp.shared_experts.gate_proj.weight"
    )
    assert not is_packed_expert_weight_key(k)


def test_per_expert_linear_layout_skips_packed_cache(tmp_path):
    from raiden.expert_nf4_cache import expert_storage_layout

    keys = [
        "model.language_model.layers.10.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.10.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.10.mlp.experts.0.down_proj.weight",
        "model.language_model.layers.10.mlp.shared_experts.gate_proj.weight",
    ]
    model_dir = tmp_path / "model"
    _fake_index(model_dir, keys)
    assert expert_storage_layout(model_dir) == "per_expert_linear"
    miss = expert_cache_preflight(model_dir, tmp_path / "cache", "nf4_freeze")
    assert miss["cache"] == "NOT_REQUIRED"
    assert miss["bf16_expert_read"] == "BNB_LINEAR4BIT"
    refuse_unready_expert_cache(miss)


def test_estimated_nf4_cache_is_volume_not_image_scale():
    forty = estimated_nf4_cache_bytes(40)
    assert 100 * 1024**3 < forty < 200 * 1024**3


def _fake_index(model_dir, keys):
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {k: "a.safetensors" for k in keys}
    weight_map["other.weight"] = "a.safetensors"
    import json

    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def _put_tensor(cache, key, payload=b"x", shape=(288, 4096, 4096)):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache_file_for_key(cache, key)
    path.write_bytes(payload)
    return {
        "shape": list(shape),
        "file": path.name,
        "bytes": path.stat().st_size,
    }


def test_preflight_states(tmp_path):
    k0 = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    k1 = "model.language_model.layers.0.mlp.experts.down_proj"
    model_dir = tmp_path / "model"
    _fake_index(model_dir, [k0, k1])
    cache = tmp_path / "cache"

    miss = expert_cache_preflight(model_dir, cache, "nf4_freeze")
    assert miss["cache"] == "MISSING"
    assert miss["bf16_expert_read"] == "LIVE_QUANT"

    info0 = _put_tensor(cache, k0)
    save_manifest(
        cache,
        {"v": 2, "status": "MATERIALIZING", "tensors": {k0: info0}, "checksum": None},
    )
    mat = expert_cache_preflight(model_dir, cache, "nf4_freeze")
    assert mat["cache"] == "MATERIALIZING"
    assert mat["bf16_expert_read"] == "LIVE_QUANT"
    assert inspect_cache_state(model_dir, cache) == "MATERIALIZING"

    save_manifest(cache, {"v": 2, "tensors": {k0: info0}})
    incomplete = expert_cache_preflight(model_dir, cache, "nf4_freeze")
    assert incomplete["cache"] == "INCOMPLETE"

    info1 = _put_tensor(cache, k1, payload=b"yy")
    tensors = {k0: info0, k1: info1}
    save_manifest(
        cache,
        {
            "v": 2,
            "status": "READY",
            "quant": "nf4",
            "layers": 1,
            "experts_per_layer": 288,
            "tensors": tensors,
            "checksum": manifest_fingerprint(tensors),
        },
    )
    hit = expert_cache_preflight(model_dir, cache, "nf4_freeze")
    assert hit["cache"] == "READY"
    assert hit["bf16_expert_read"] == "SKIPPED"
    assert hit["cached_experts"] == 2
    assert hit["layers"] == 1
    assert hit["experts_per_layer"] == 288

    bad = dict(info1)
    bad["bytes"] = 999999
    save_manifest(
        cache,
        {
            "v": 2,
            "status": "READY",
            "tensors": {k0: info0, k1: bad},
            "checksum": manifest_fingerprint({k0: info0, k1: bad}),
        },
    )
    corrupted = expert_cache_preflight(model_dir, cache, "nf4_freeze")
    assert corrupted["cache"] == "CORRUPTED"
    assert corrupted["bf16_expert_read"] == "LIVE_QUANT"


def test_refuse_unready_cache(tmp_path, monkeypatch):
    from raiden.compatibility import RaidenCompatibilityError

    k = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    model_dir = tmp_path / "model"
    _fake_index(model_dir, [k])
    pre = expert_cache_preflight(model_dir, tmp_path / "cache", "nf4_freeze")
    try:
        refuse_unready_expert_cache(pre)
        raise AssertionError("expected refuse")
    except RaidenCompatibilityError as exc:
        assert "MISSING" in str(exc)
    monkeypatch.setenv("RAIDEN_ALLOW_LIVE_EXPERT_QUANT", "1")
    refuse_unready_expert_cache(pre)


def test_v1_complete_manifest_is_ready(tmp_path):
    key = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    model_dir = tmp_path / "model"
    _fake_index(model_dir, [key])
    cache = tmp_path / "cache"
    info = _put_tensor(cache, key)
    save_manifest(cache, {"v": 1, "tensors": {key: info}})
    hit = expert_cache_preflight(model_dir, cache, "nf4_freeze")
    assert hit["cache"] == "READY"
    assert hit["bf16_expert_read"] == "SKIPPED"


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


def test_validate_runtime_cache_checks(tmp_path, monkeypatch):
    from raiden.validate_runtime import cache_checks

    key = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    model_dir = tmp_path / "model"
    cache = tmp_path / "cache"
    _fake_index(model_dir, [key])
    info = _put_tensor(cache, key)
    tensors = {key: info}
    save_manifest(
        cache,
        {
            "v": 2,
            "status": "READY",
            "tensors": tensors,
            "checksum": manifest_fingerprint(tensors),
            "layers": 1,
            "experts_per_layer": 288,
        },
    )
    monkeypatch.setenv("RAIDEN_EXPERT_NF4_CACHE", str(cache))
    checks, pre = cache_checks(str(model_dir), "nf4_freeze")
    assert pre["cache"] == "READY"
    assert all(ok for _, ok, _ in checks)


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

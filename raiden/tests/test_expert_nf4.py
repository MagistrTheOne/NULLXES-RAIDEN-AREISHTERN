from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.expert_nf4 import (
    VRAM_BF16_EXPERTS_MIN_BYTES,
    count_nf4_expert_modules,
    nf4_expert_chunk_size,
    resolve_packed_expert_policy,
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

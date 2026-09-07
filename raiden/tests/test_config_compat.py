from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.compatibility import (
    RaidenCompatibilityError,
    _is_native_fp8,
    _probe_bitsandbytes_cuda,
    check_environment,
)
from raiden.config import RaidenConfig
from raiden.resume import resolve_resume


def test_fp8_detector():
    assert _is_native_fp8({"quantization_config": {"fmt": "e4m3", "activation_scheme": "dynamic"}})
    assert not _is_native_fp8({"quantization_config": {}})


def test_yaml_loads(tmp_path):
    src = Path(__file__).resolve().parents[1] / "configs" / "raiden_qlora.yaml"
    cfg = RaidenConfig.from_yaml(src)
    assert cfg.qlora.method == "qlora"
    assert cfg.qlora.bits == 4
    assert cfg.lora.r == 64
    assert cfg.freeze.freeze_router is True
    assert cfg.freeze.router_experiment_enabled is False
    assert cfg.train.max_length in (4096, 8192)
    assert cfg.dataset.mix["identity"] == 0.07
    assert cfg.dataset.mix["agency"] == 0.08
    assert cfg.dataset.mix["hard_judgment"] == 0.07
    assert cfg.dataset.mix["pressure_resistance"] == 0.07
    assert cfg.dataset.mix["banking"] == 0.06
    assert abs(sum(cfg.dataset.mix.values()) - 1.0) < 1e-9


def test_resume_auto_none(tmp_path):
    assert resolve_resume("none", str(tmp_path)) is None
    ck = tmp_path / "checkpoint-200"
    ck.mkdir()
    (ck / "adapter_config.json").write_text("{}", encoding="utf-8")
    found = resolve_resume("auto", str(tmp_path))
    assert found.endswith("checkpoint-200")


def test_bnb_probe_does_not_use_removed_compiled_with_cuda_symbol():
    import inspect

    from raiden import compatibility as compat

    source = inspect.getsource(compat)
    assert "from bitsandbytes.cextension import COMPILED_WITH_CUDA" not in source
    ok, fact = _probe_bitsandbytes_cuda()
    assert isinstance(ok, bool)
    assert isinstance(fact, str)


def test_forbidden_crack_base_is_hard_error():
    from raiden.family import forbidden_base_hit

    assert forbidden_base_hit("dealignai/GLM-5.3-CYBERSECURITY-FP8")
    assert forbidden_base_hit("dealignai/GLM-5.3-UNCENSORED-FP8")
    assert forbidden_base_hit("JANGQ-AI/GLM-5.3-FP8")
    assert forbidden_base_hit("zai-org/GLM-5.3-Flash-BF16") is None
    report = check_environment(base_model="dealignai/GLM-5.3-CYBERSECURITY-FP8")
    assert report.ok is False
    assert any("forbidden" in e.lower() for e in report.errors)


def test_family_yaml_has_two_tracks():
    import yaml

    src = Path(__file__).resolve().parents[1] / "configs" / "family.yaml"
    family = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert family["tracks"]["flash"]["train_weights"] == "zai-org/GLM-5.3-Flash-BF16"
    assert family["tracks"]["glm53"]["train_weights"] == "zai-org/GLM-5.3-BF16"
    assert family["tracks"]["flash"]["status"] == "active_stage1"
    assert family["tracks"]["glm53"]["status"] == "specified_not_wired"
    assert "dealignai/GLM-5.3-CYBERSECURITY-FP8" in family["forbidden_bases"]


def test_unsloth_backend_is_hard_error():
    local = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dummy_model"
    report = check_environment(base_model=str(local), backend="unsloth", refuse_native_fp8=False)
    assert report.ok is False
    assert any("unsloth" in e.lower() for e in report.errors)

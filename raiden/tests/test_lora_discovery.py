from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from raiden.lora import census_from_model, plan_stage1_lora


class FakeRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8, 16))


class FakeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(8, 32, 16))
        self.down_proj = nn.Parameter(torch.zeros(8, 16, 32))


class FakeMLP(nn.Module):
    def __init__(self, moe=True):
        super().__init__()
        if moe:
            self.experts = FakeExperts()
            self.gate = FakeRouter()
            self.shared_experts = nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(16, 32, bias=False),
                    "up_proj": nn.Linear(16, 32, bias=False),
                    "down_proj": nn.Linear(32, 16, bias=False),
                }
            )
        else:
            self.gate_proj = nn.Linear(16, 32, bias=False)
            self.up_proj = nn.Linear(16, 32, bias=False)
            self.down_proj = nn.Linear(32, 16, bias=False)


class FakeAttn(nn.Module):
    def __init__(self, kind="linear"):
        super().__init__()
        if kind == "linear":
            self.q_proj = nn.Linear(16, 16, bias=False)
            self.k_proj = nn.Linear(16, 16, bias=False)
            self.v_proj = nn.Linear(16, 16, bias=False)
            self.o_proj = nn.Linear(16, 16, bias=False)
        else:
            self.q_a_proj = nn.Linear(16, 8, bias=False)
            self.q_b_proj = nn.Linear(8, 16, bias=False)
            self.kv_a_proj_with_mqa = nn.Linear(16, 8, bias=False)
            self.kv_b_proj = nn.Linear(8, 16, bias=False)
            self.o_proj = nn.Linear(16, 16, bias=False)
            self.indexer = nn.ModuleDict(
                {"wq_b": nn.Linear(8, 16, bias=False), "wk": nn.Linear(16, 8, bias=False)}
            )


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 48, bias=False)
        self.proj = nn.Linear(16, 16, bias=False)
        self.gate_proj = nn.Linear(16, 32, bias=False)


class FakeGlm5(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["Glm5NextForConditionalGeneration"],
            model_type="glm5_next",
            text_config=SimpleNamespace(
                num_hidden_layers=4,
                layer_types=["linear_attention", "linear_attention", "deepseek_sparse_attention", "linear_attention"],
                mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
            ),
        )
        self.visual = FakeVision()
        self.language_model = nn.ModuleDict({})
        layers = nn.ModuleList()
        for i in range(4):
            block = nn.ModuleDict(
                {
                    "self_attn": FakeAttn("linear" if i != 2 else "sparse"),
                    "mlp": FakeMLP(moe=(i != 0)),
                }
            )
            layers.append(block)
        self.language_model.layers = layers
        self.lm_head = nn.Linear(16, 32, bias=False)


def test_stage1_plan_excludes_vision_router_experts_indexer():
    model = FakeGlm5()
    plan = plan_stage1_lora(model)
    assert "q_proj" in plan.target_modules
    assert "o_proj" in plan.target_modules
    assert "gate_proj" in plan.target_modules
    assert "down_proj" in plan.target_modules
    joined = " ".join(plan.target_full_paths)
    assert "visual" not in joined
    assert ".gate." not in joined and not any(p.endswith(".gate") for p in plan.target_full_paths)
    assert ".experts." not in joined
    assert "indexer" not in joined
    assert "lm_head" not in joined
    census = census_from_model(model)
    assert census.architecture == "Glm5NextForConditionalGeneration"
    assert census.n_layers == 4

#!/usr/bin/env python3
"""Inspect GLM-5.3-Flash architecture and propose Stage I LoRA targets.

Default: config-only (no weight download).
--full: load the model (RunPod) and walk named_modules().
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.compatibility import check_environment, parse_hf_config
from raiden.config import RaidenConfig
from raiden.identity import BASE_REPO_BF16
from raiden.paths import apply_cache_env, logs_dir


def config_only(model_id: str) -> dict:
    cfg = parse_hf_config(model_id, token=os.environ.get("HF_TOKEN"))
    text = cfg.get("text_config") or {}
    vision = cfg.get("vision_config") or {}
    layer_types = text.get("layer_types") or []
    mlp_types = text.get("mlp_layer_types") or []
    linear_n = sum(1 for t in layer_types if t == "linear_attention")
    sparse_n = sum(1 for t in layer_types if t == "deepseek_sparse_attention")
    dense_mlp = sum(1 for t in mlp_types if t == "dense")
    sparse_mlp = sum(1 for t in mlp_types if t == "sparse")
    return {
        "model_id": model_id,
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "transformers_version_in_card": cfg.get("transformers_version"),
        "native_quantization_config": cfg.get("quantization_config") is not None,
        "quantization_fmt": (cfg.get("quantization_config") or {}).get("fmt"),
        "text": {
            "hidden_size": text.get("hidden_size"),
            "num_hidden_layers": text.get("num_hidden_layers"),
            "num_attention_heads": text.get("num_attention_heads"),
            "n_routed_experts": text.get("n_routed_experts"),
            "n_shared_experts": text.get("n_shared_experts"),
            "num_experts_per_tok": text.get("num_experts_per_tok"),
            "first_k_dense_replace": text.get("first_k_dense_replace"),
            "moe_intermediate_size": text.get("moe_intermediate_size"),
            "q_lora_rank": text.get("q_lora_rank"),
            "kv_lora_rank": text.get("kv_lora_rank"),
            "mhc": text.get("mhc"),
            "max_position_embeddings": text.get("max_position_embeddings"),
            "linear_attention_layers": linear_n,
            "sparse_attention_layers": sparse_n,
            "dense_mlp_layers": dense_mlp,
            "moe_mlp_layers": sparse_mlp,
            "layer_types": layer_types,
            "mlp_layer_types": mlp_types,
        },
        "vision": {
            "present": bool(vision),
            "model_type": vision.get("model_type"),
            "depth": vision.get("depth"),
            "hidden_size": vision.get("hidden_size"),
            "image_size": vision.get("image_size"),
            "patch_size": vision.get("patch_size"),
        },
        "candidate_lora_linear_names_from_source": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
            "b_proj",
            "fused_qkvbfg_a_proj",
        ],
        "router_modules_expected": [
            "mlp.gate (Glm5NextTextTopkRouter) — nn.Parameter weight, FREEZE Stage I"
        ],
        "expert_modules_expected": [
            "mlp.experts.gate_up_proj / down_proj — packed nn.Parameter, NOT nn.Linear, freeze Stage I"
        ],
        "vision_modules_expected": ["model.visual (Glm5NextVisionModel) — freeze Stage I"],
        "notes": [
            "Do not assume Llama module names. Confirm with --full on RunPod.",
            "Default Hub dump zai-org/GLM-5.3-Flash is native FP8. QLoRA uses BF16 weights.",
            "bitsandbytes Linear4bit will not convert packed expert Parameters.",
        ],
    }


def main() -> int:
    apply_cache_env()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--config", default="configs/raiden_qlora.yaml")
    p.add_argument("--full", action="store_true", help="Load weights and walk named_modules()")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model_id = args.model
    if model_id is None:
        if Path(args.config).exists():
            model_id = RaidenConfig.from_yaml(args.config).base_model
        else:
            model_id = BASE_REPO_BF16

    report = check_environment(base_model=model_id, backend="hf_qlora")
    payload = {
        "compat_ok": report.ok,
        "compat_errors": report.errors,
        "compat_warnings": report.warnings,
        "compat_facts": report.facts,
        "config_only": config_only(model_id),
    }

    if args.full:
        from raiden.config import RaidenConfig as RC
        from raiden.lora import census_from_model, dump_census_json, plan_stage1_lora
        from raiden.model import load_qlora_model

        cfg = RC.from_yaml(args.config) if Path(args.config).exists() else None
        if cfg is None:
            raise SystemExit(" --full requires configs/raiden_qlora.yaml")
        cfg.base_model = model_id
        model, plan, census = load_qlora_model(cfg, inspection_out=None)
        payload["named_modules_census"] = json.loads(dump_census_json(census, plan))
        del model

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    out = Path(args.out) if args.out else (logs_dir() / "inspect_model.json")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    except Exception:
        Path("inspect_model.json").write_text(text + "\n", encoding="utf-8")
        print("wrote ./inspect_model.json")
    print(text)
    return 0 if report.ok or args.full is False else 1


if __name__ == "__main__":
    raise SystemExit(main())

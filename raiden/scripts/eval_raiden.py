#!/usr/bin/env python3
"""Run RAIDEN eval suite against a checkpoint (adapter + base)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.config import RaidenConfig
from raiden.eval.runner import run_full_eval
from raiden.logging import setup_logging
from raiden.paths import apply_cache_env, logs_dir


def load_adapter_for_eval(cfg, adapter: str):
    from peft import PeftModel, prepare_model_for_kbit_training

    from raiden.freeze import apply_freeze
    from raiden.model import load_quantized_base, load_tokenizer_and_processor

    tokenizer, _ = load_tokenizer_and_processor(cfg.base_model, token=os.environ.get("HF_TOKEN"))
    base = load_quantized_base(cfg)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter)
    apply_freeze(model, cfg.freeze)
    model.eval()
    return model, tokenizer


def main() -> int:
    apply_cache_env()
    log = setup_logging("raiden-eval")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/raiden_qlora.yaml")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--adapter", default=None, help="PEFT adapter dir; default: latest checkpoint")
    p.add_argument("--out", default=None)
    p.add_argument("--base-only", action="store_true", help="eval frozen base (no RAIDEN adapter)")
    args = p.parse_args()

    cfg = RaidenConfig.from_yaml(args.config)
    adapter = args.adapter
    if adapter is None and not args.base_only:
        from raiden.paths import latest_checkpoint

        found = latest_checkpoint(Path(cfg.train.output_dir))
        adapter = str(found) if found else cfg.train.output_dir
        log.info("using adapter %s", adapter)

    if args.base_only:
        from raiden.model import load_tokenizer_and_processor
        from transformers import AutoModelForImageTextToText

        tokenizer, _ = load_tokenizer_and_processor(cfg.base_model, token=os.environ.get("HF_TOKEN"))
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                cfg.base_model,
                device_map="auto",
                torch_dtype="auto",
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model,
                device_map="auto",
                torch_dtype="auto",
                token=os.environ.get("HF_TOKEN"),
            )
    else:
        model, tokenizer = load_adapter_for_eval(cfg, adapter)

    out = Path(args.out) if args.out else (logs_dir() / "eval_report.json")
    report = run_full_eval(model, tokenizer, cfg.eval_dir, out)
    log.info(
        "identity=%s sycophancy=%s residue=%s behavior=%s/%s retention=%s",
        report.get("identity_score"),
        report.get("sycophancy_score"),
        report.get("assistant_residue_score", report.get("tone_score")),
        report.get("raiden_behavior"),
        report.get("raiden_behavior_score"),
        report.get("retention_score"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

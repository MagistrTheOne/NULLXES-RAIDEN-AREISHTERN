"""CLI: RAIDEN Stage I QLoRA SFT. RunPod only. Never starts a dummy LoRA-bf16 fallback."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from raiden.compatibility import RaidenCompatibilityError
from raiden.config import RaidenConfig
from raiden.expert_nf4 import resolve_packed_expert_policy
from raiden.expert_nf4_cache import expert_cache_preflight, expert_nf4_cache_dir, refuse_unready_expert_cache
from raiden.logging import setup_logging, write_json
from raiden.model import load_qlora_model, load_tokenizer_and_processor
from raiden.paths import apply_cache_env, logs_dir
from raiden.trainer import build_trainer, load_jsonl_dataset, train_or_resume


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RAIDEN Stage I QLoRA SFT")
    p.add_argument("--config", default="configs/raiden_qlora.yaml")
    p.add_argument("--resume", default=None, help="override resume: auto|none|path")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--train-path", default=None)
    p.add_argument("--val-path", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    apply_cache_env()
    args = parse_args(argv)
    logger = setup_logging("raiden-train")
    cfg = RaidenConfig.from_yaml(args.config)
    if args.resume is not None:
        cfg.train.resume_from_checkpoint = args.resume
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.train_path:
        cfg.dataset.train_path = args.train_path
    if args.val_path:
        cfg.dataset.val_path = args.val_path

    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(cfg.train.output_dir) / "raiden_resolved_config.json", cfg.raw)

    if not Path(cfg.dataset.train_path).exists():
        logger.error("train jsonl missing: %s — run scripts/prepare_dataset.py first", cfg.dataset.train_path)
        return 2
    if not Path(cfg.dataset.val_path).exists():
        logger.error("val jsonl missing: %s", cfg.dataset.val_path)
        return 2

    try:
        policy = resolve_packed_expert_policy(cfg.qlora.packed_expert_policy)
        if policy == "nf4_freeze":
            pre = expert_cache_preflight(cfg.base_model, expert_nf4_cache_dir(), policy)
            refuse_unready_expert_cache(pre)
            logger.info("runtime validate cache=%s bf16_expert_read=%s", pre["cache"], pre["bf16_expert_read"])
        tokenizer, _processor = load_tokenizer_and_processor(
            cfg.base_model, token=os.environ.get("HF_TOKEN")
        )
        model, plan, census = load_qlora_model(
            cfg, inspection_out=str(logs_dir() / "inspect_at_train.json")
        )
        train_ds = load_jsonl_dataset(cfg.dataset.train_path)
        val_ds = load_jsonl_dataset(cfg.dataset.val_path)
        logger.info("datasets: train=%s val=%s", len(train_ds), len(val_ds))
        logger.info("STARTING RAIDEN SFT STAGE I")

        def _eval_fn(model, tokenizer, step):
            from raiden.eval.runner import lightweight_eval

            return lightweight_eval(model, tokenizer, step=step, cfg=cfg)

        trainer = build_trainer(cfg, model, tokenizer, train_ds, val_ds, eval_fn=_eval_fn)
        train_or_resume(trainer, cfg)
        trainer.save_model(cfg.train.output_dir)
        tokenizer.save_pretrained(cfg.train.output_dir)
        write_json(
            Path(cfg.train.output_dir) / "lora_plan.json",
            {
                "target_modules": plan.target_modules,
                "n_target_paths": len(plan.target_full_paths),
                "notes": plan.notes,
                "architecture": census.architecture,
            },
        )
        logger.info("training finished; adapter saved to %s", cfg.train.output_dir)
        return 0
    except RaidenCompatibilityError as exc:
        logger.error("%s", exc)
        return 3
    except Exception:
        logger.exception("training crashed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""TRL SFTTrainer construction for RAIDEN Stage I QLoRA."""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

from raiden.callbacks import (
    EvalCurveCallback,
    LightweightRaidenEvalCallback,
    SigtermCheckpointCallback,
)
from raiden.logging import choose_report_to
from raiden.resume import resolve_resume

logger = logging.getLogger("raiden")


def _sft_config_class():
    from trl import SFTConfig

    return SFTConfig


def _filter_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters)
    if "kwargs" in accepted or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        # still drop known-unknowns that older/newer TRL renamed
        pass
    out = {}
    dropped = []
    for k, v in kwargs.items():
        if k in accepted or "kwargs" in accepted:
            out[k] = v
        else:
            dropped.append(k)
    if dropped:
        logger.warning("dropping unsupported SFTConfig keys: %s", dropped)
    return out


def build_sft_args(cfg, tokenizer) -> Any:
    SFTConfig = _sft_config_class()
    report_to = choose_report_to(cfg.train.report_to)
    kwargs = dict(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        eval_strategy=cfg.train.eval_strategy,
        eval_steps=cfg.train.eval_steps,
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs=cfg.train.gradient_checkpointing_kwargs,
        optim=cfg.train.optim,
        max_grad_norm=cfg.train.max_grad_norm,
        seed=cfg.train.seed,
        data_seed=cfg.train.data_seed,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        dataloader_pin_memory=cfg.train.dataloader_pin_memory,
        report_to=report_to,
        run_name=cfg.train.run_name,
        remove_unused_columns=cfg.train.remove_unused_columns,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        max_length=cfg.train.max_length,
        packing=cfg.train.packing,
        dataset_kwargs={"skip_prepare_dataset": False},
        logging_dir=os.path.join(os.environ.get("RAIDEN_LOGS", "/workspace/logs"), "tb"),
        push_to_hub=cfg.train.push_to_hub,
        hub_model_id=cfg.train.hub_model_id,
        hub_private_repo=cfg.train.hub_private_repo,
        save_safetensors=True,
        logging_first_step=True,
    )
    # assistant-only / completion-only: name differs across TRL versions
    for key in ("assistant_only_loss", "completion_only_loss"):
        trial = dict(kwargs)
        trial[key] = cfg.train.assistant_only_loss
        filtered = _filter_kwargs(SFTConfig, trial)
        if key in filtered:
            kwargs = filtered
            break
    else:
        kwargs = _filter_kwargs(SFTConfig, kwargs)
    return SFTConfig(**kwargs)


def load_jsonl_dataset(path: str):
    from datasets import load_dataset

    return load_dataset("json", data_files=path, split="train")


def build_trainer(cfg, model, tokenizer, train_ds, eval_ds, eval_fn=None):
    from trl import SFTTrainer

    args = build_sft_args(cfg, tokenizer)
    callbacks = [
        SigtermCheckpointCallback(),
        EvalCurveCallback(),
        LightweightRaidenEvalCallback(eval_fn=eval_fn),
    ]
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
    )
    # TRL renamed tokenizer -> processing_class
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "formatting_func" in sig.parameters:
        # messages column is natively supported in recent TRL when dataset has "messages"
        pass
    trainer = SFTTrainer(**trainer_kwargs)
    return trainer


def train_or_resume(trainer, cfg) -> Any:
    resume = resolve_resume(cfg.train.resume_from_checkpoint, cfg.train.output_dir)
    logger.info("calling trainer.train(resume_from_checkpoint=%s)", resume)
    return trainer.train(resume_from_checkpoint=resume)

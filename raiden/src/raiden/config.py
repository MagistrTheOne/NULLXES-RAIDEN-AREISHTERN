"""YAML config loader for RAIDEN Stage I QLoRA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _get(d: dict, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass
class QLoRAConfig:
    method: str = "qlora"
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    quant_storage: str = "bfloat16"
    backend: str = "hf_qlora"  # hf_qlora | unsloth
    require_linear_4bit: bool = True
    require_expert_4bit: bool = False
    packed_expert_policy: str = "nf4_freeze"
    refuse_native_fp8: bool = True
    allow_fp8_dequant: bool = False
    skip_modules: tuple[str, ...] = ("visual", "lm_head")


@dataclass
class LoRAConfigSpec:
    r: int = 64
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_mode: str = "auto_stage1"  # auto_stage1 | explicit | all_linear_filtered
    explicit_targets: tuple[str, ...] = ()
    modules_to_save: tuple[str, ...] = ()


@dataclass
class FreezeConfig:
    freeze_router: bool = True
    freeze_vision: bool = True
    freeze_packed_experts: bool = True
    freeze_indexer: bool = True
    freeze_hyper_connections: bool = True
    freeze_embeddings: bool = True
    freeze_lm_head: bool = True
    router_experiment_enabled: bool = False
    router_experiment_train_router: bool = False


@dataclass
class TrainConfig:
    output_dir: str = "/workspace/checkpoints/raiden-sft-stage1"
    max_length: int = 4096
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 5
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 20
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 1.0
    seed: int = 42
    data_seed: int = 42
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    report_to: str = "auto"
    run_name: str = "raiden-sft-stage1"
    packing: bool = False
    assistant_only_loss: bool = True
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    distributed_strategy: str = "device_map_auto"
    resume_from_checkpoint: str | None = "auto"
    push_to_hub: bool = False
    hub_model_id: str = "NULLXES/RAIDEN-AREISHTERN-v0.1"
    hub_private_repo: bool = True


@dataclass
class DatasetConfig:
    train_path: str = "/workspace/datasets/raiden/train.jsonl"
    val_path: str = "/workspace/datasets/raiden/validation.jsonl"
    preference_path: str = "/workspace/datasets/raiden/preference.jsonl"
    target_train_size: int = 24000
    target_val_size: int = 2000
    replay_size: int = 0
    languages: tuple[str, ...] = ("ru", "en", "mixed")
    mix: dict[str, float] = field(
        default_factory=lambda: {
            "identity": 0.07,
            "style": 0.08,
            "anti_sycophancy": 0.07,
            "direct_decision": 0.06,
            "personal_maga": 0.05,
            "uncertainty": 0.08,
            "identity_attack": 0.04,
            "capability_replay": 0.07,
            "agency": 0.08,
            "pressure_resistance": 0.07,
            "hard_judgment": 0.07,
            "command_execution": 0.05,
            "evidence_reversal": 0.05,
            "social_boundary": 0.02,
            "register_control": 0.02,
            "independence": 0.06,
            "banking": 0.06,
        }
    )


@dataclass
class RaidenConfig:
    raw: dict
    base_model: str
    product_model_id: str
    qlora: QLoRAConfig
    lora: LoRAConfigSpec
    freeze: FreezeConfig
    train: TrainConfig
    dataset: DatasetConfig
    eval_dir: str
    chat_template_kwargs: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RaidenConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        q = raw.get("qlora", {}) or {}
        l = raw.get("lora", {}) or {}
        fr = raw.get("freeze", {}) or {}
        tr = raw.get("train", {}) or {}
        ds = raw.get("dataset", {}) or {}
        qlora = QLoRAConfig(
            method=q.get("method", "qlora"),
            bits=int(q.get("bits", 4)),
            quant_type=q.get("quant_type", "nf4"),
            double_quant=bool(q.get("double_quant", True)),
            compute_dtype=q.get("compute_dtype", "bfloat16"),
            quant_storage=q.get("quant_storage", "bfloat16"),
            backend=q.get("backend", "hf_qlora"),
            require_linear_4bit=bool(q.get("require_linear_4bit", True)),
            require_expert_4bit=bool(q.get("require_expert_4bit", False)),
            packed_expert_policy=q.get("packed_expert_policy", "nf4_freeze"),
            refuse_native_fp8=bool(q.get("refuse_native_fp8", True)),
            allow_fp8_dequant=bool(q.get("allow_fp8_dequant", False)),
            skip_modules=tuple(q.get("skip_modules", ["visual", "lm_head"])),
        )
        lora = LoRAConfigSpec(
            r=int(l.get("r", 64)),
            alpha=int(l.get("alpha", l.get("r", 64))),
            dropout=float(l.get("dropout", 0.05)),
            bias=l.get("bias", "none"),
            task_type=l.get("task_type", "CAUSAL_LM"),
            target_mode=l.get("target_mode", "auto_stage1"),
            explicit_targets=tuple(l.get("explicit_targets", []) or []),
            modules_to_save=tuple(l.get("modules_to_save", []) or []),
        )
        freeze = FreezeConfig(
            freeze_router=bool(fr.get("freeze_router", True)),
            freeze_vision=bool(fr.get("freeze_vision", True)),
            freeze_packed_experts=bool(fr.get("freeze_packed_experts", True)),
            freeze_indexer=bool(fr.get("freeze_indexer", True)),
            freeze_hyper_connections=bool(fr.get("freeze_hyper_connections", True)),
            freeze_embeddings=bool(fr.get("freeze_embeddings", True)),
            freeze_lm_head=bool(fr.get("freeze_lm_head", True)),
            router_experiment_enabled=bool(_get(fr, "router_experiment", "enabled", default=False)),
            router_experiment_train_router=bool(
                _get(fr, "router_experiment", "train_router", default=False)
            ),
        )
        train = TrainConfig(
            output_dir=tr.get("output_dir", "/workspace/checkpoints/raiden-sft-stage1"),
            max_length=int(tr.get("max_length", 4096)),
            num_train_epochs=float(tr.get("num_train_epochs", 1.0)),
            max_steps=int(tr.get("max_steps", -1)),
            per_device_train_batch_size=int(tr.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(tr.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 16)),
            learning_rate=float(tr.get("learning_rate", 1.0e-4)),
            lr_scheduler_type=tr.get("lr_scheduler_type", "cosine"),
            warmup_ratio=float(tr.get("warmup_ratio", 0.03)),
            weight_decay=float(tr.get("weight_decay", 0.0)),
            logging_steps=int(tr.get("logging_steps", 5)),
            eval_strategy=tr.get("eval_strategy", "steps"),
            eval_steps=int(tr.get("eval_steps", 100)),
            save_strategy=tr.get("save_strategy", "steps"),
            save_steps=int(tr.get("save_steps", 100)),
            save_total_limit=int(tr.get("save_total_limit", 20)),
            load_best_model_at_end=bool(tr.get("load_best_model_at_end", False)),
            bf16=bool(tr.get("bf16", True)),
            fp16=bool(tr.get("fp16", False)),
            gradient_checkpointing=bool(tr.get("gradient_checkpointing", True)),
            optim=tr.get("optim", "paged_adamw_8bit"),
            max_grad_norm=float(tr.get("max_grad_norm", 1.0)),
            seed=int(tr.get("seed", 42)),
            data_seed=int(tr.get("data_seed", 42)),
            dataloader_num_workers=int(tr.get("dataloader_num_workers", 2)),
            report_to=tr.get("report_to", "auto"),
            run_name=tr.get("run_name", "raiden-sft-stage1"),
            packing=bool(tr.get("packing", False)),
            assistant_only_loss=bool(tr.get("assistant_only_loss", True)),
            distributed_strategy=tr.get("distributed_strategy", "device_map_auto"),
            resume_from_checkpoint=tr.get("resume_from_checkpoint", "auto"),
            push_to_hub=bool(tr.get("push_to_hub", False)),
            hub_model_id=tr.get("hub_model_id", "NULLXES/RAIDEN-AREISHTERN-v0.1"),
            hub_private_repo=bool(tr.get("hub_private_repo", True)),
        )
        dataset = DatasetConfig(
            train_path=ds.get("train_path", "/workspace/datasets/raiden/train.jsonl"),
            val_path=ds.get("val_path", "/workspace/datasets/raiden/validation.jsonl"),
            preference_path=ds.get("preference_path", "/workspace/datasets/raiden/preference.jsonl"),
            target_train_size=int(ds.get("target_train_size", 24000)),
            target_val_size=int(ds.get("target_val_size", 2000)),
            replay_size=int(ds.get("replay_size", 0)),
            languages=tuple(ds.get("languages", ["ru", "en", "mixed"])),
            mix=ds.get("mix") or DatasetConfig().mix,
        )
        base_model = raw.get("base_model", "zai-org/GLM-5.3-Flash-BF16")
        from raiden.expert_nf4_cache import resolve_model_dir

        resolved = resolve_model_dir(base_model)
        if (resolved / "model.safetensors.index.json").is_file():
            base_model = str(resolved)
        return cls(
            raw=raw,
            base_model=base_model,
            product_model_id=raw.get("product_model_id", "raiden-areishtern"),
            qlora=qlora,
            lora=lora,
            freeze=freeze,
            train=train,
            dataset=dataset,
            eval_dir=raw.get("eval_dir", "/workspace/datasets/raiden/eval"),
            chat_template_kwargs=raw.get("chat_template_kwargs")
            or {"reasoning_effort": "low", "clear_thinking": True},
        )

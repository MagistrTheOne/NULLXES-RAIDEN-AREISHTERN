# Training — Stage I QLoRA

Stage I is **behavioral / identity post-training** with **QLoRA only**.

It does not rebuild coding, math, tools, or world knowledge. It does not fine-tune the MoE router.

Operator commands: [RUNPOD.md](RUNPOD.md).  
Behavior target: [IDENTITY.md](IDENTITY.md).

## Method

- Parameter-efficient SFT with 4-bit NF4 (bitsandbytes) + LoRA.
- Training weights: BF16 checkpoint of the foundation model. The public FP8 Hub dump is **not** a QLoRA base.
- LoRA targets are discovered from `named_modules()` (`scripts/inspect_model.py`). No Llama name assumptions.
- If 4-bit Linear conversion fails, training **stops** (`RAIDEN_QLORA_INCOMPATIBLE`). There is no silent fallback to LoRA-bf16 or full-FT.
- Unsloth is rejected for Stage I: its MoE path would change method.

Default LoRA: `r=64`, `alpha=64`, `dropout=0.05`, `max_length=4096`.

Config: `configs/raiden_qlora.yaml`. Preference / DPO config exists and is **disabled** until an SFT checkpoint is chosen from the eval curve.

## Freeze policy (Stage I)

| Component | Policy |
| --------- | ------ |
| MoE router / gate | Frozen |
| Packed expert parameters | Frozen (NF4 on single-B300; BF16 cannot fit 275 GiB) |
| Vision encoder | Frozen |
| Indexer, mHC, embeddings, lm_head | Frozen |
| Linear attention + dense/shared MLP | LoRA |

`router_experiment.enabled` is false. Enabling it is a later research run, not the baseline.

## Dataset mix — RAIDEN HARD PRIOR

Not `helpfulness − politeness − sycophancy`. Axes: agency, dominance, judgment, pressure resistance, epistemic discipline.

Target shares live in `DEFAULT_MIX`. Small unique pools (identity) are **not** cloned. Leftover unique mass from expandable axes fills the count. If unique < requested, training data generation aborts with `RAIDEN_DATASET_INSUFFICIENT_DIVERSITY`.

Banking is in the mix as a risk/judgment domain, not a consultant voice. Eval is the behavioral vector in [EVALUATION.md](EVALUATION.md), not `tone_score` alone.

See [DATASET.md](DATASET.md).

Schema: `data/schemas/sft.schema.json`.  
Generator: `src/raiden/data_gen/synthesize.py`.  
Validator: `scripts/validate_dataset.py`.

Languages: Russian, English, mixed. Empty-system rows exist so identity is not prompt-only.

Preference pairs are hard negatives (correct-but-deferential, stubbornness vs update). `train_raiden.sh` does not start DPO.

## Compatibility notes (foundation)

The Stage I foundation is a hybrid MoE multimodal decoder (`Glm5NextForConditionalGeneration` in current Transformers). Packed experts are `nn.Parameter`, not `nn.Linear`, so bitsandbytes Linear4bit does not cover them. Stage I QLoRA is Linear-module QLoRA with frozen packed experts. On one B300 those experts are stored NF4 (`packed_expert_policy: nf4_freeze`) so the load fits; LoRA still does not adapt them. That is documented, not a silent method switch to LoRA-bf16. Setting `qlora.require_expert_4bit: true` still aborts (that flag means Linear4bit-on-experts, which cannot exist).

Failure table and Hub details stay in operator logs and this document — not in the public README.

## Checkpoints

Save every 100 optimizer steps for a behavioral curve: train/val loss plus identity, sycophancy, tone, retention. Pick the **earliest** checkpoint where RAIDEN behavior has moved and retention has not collapsed.

## Tests

```bash
PYTHONPATH=src pytest -q
```

Do not run GPU train/download from a workstation. See [CONTRIBUTING.md](../CONTRIBUTING.md).

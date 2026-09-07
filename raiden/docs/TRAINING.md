# Training — Stage I QLoRA

Stage I is **behavioral / identity post-training** with **QLoRA only**.

It does not rebuild coding, math, tools, or world knowledge. It does not fine-tune the MoE router.

Operator commands: [RUNPOD.md](RUNPOD.md).  
Live B300 load/cache: [RUNBOOK.md](RUNBOOK.md).  
Behavior target: [IDENTITY.md](IDENTITY.md).

## Method

- Parameter-efficient SFT with 4-bit NF4 (bitsandbytes) + LoRA.
- Training weights: BF16 checkpoint of the foundation model. The public FP8 Hub dump is **not** a QLoRA base.
- LoRA targets are discovered from `named_modules()` (`scripts/inspect_model.py`). No Llama name assumptions.
- If 4-bit Linear conversion fails, training **stops** (`RAIDEN_QLORA_INCOMPATIBLE`). There is no silent fallback to LoRA-bf16 or full-FT.
- Unsloth is rejected for Stage I: its MoE path would change method.

Default LoRA: `r=64`, `alpha=64`, `dropout=0.05`, `max_length=4096`.

Config: `configs/raiden_qlora.yaml`. Family (two SKUs): `configs/family.yaml`. Preference / DPO config exists and is **disabled** until an SFT checkpoint is chosen from the eval curve.

## Two SKUs

| Track | Public id | Train weights | Status |
| ----- | --------- | ------------- | ------ |
| Flash | `raiden-areishtern` | `zai-org/GLM-5.3-Flash-BF16` | **active Stage I** (this trainer) |
| GLM-5.3 | `raiden-areishtern-5.3` | `zai-org/GLM-5.3-BF16` | specified, **not wired** |

Flash is MIT, hybrid Glm5Next, ~320B, one B300 + `nf4_freeze`.

GLM-5.3 is a different model: `glm_moe_dsa`, ~753B / ~39B active, 78 layers, text-only, GLM-5.3 license (not MIT). Official FP8 card is `zai-org/GLM-5.3`. Serving that FP8 class fits **8× H200 TP8** (same node class [dealignai documented for a third-party FP8 dump](https://huggingface.co/dealignai/GLM-5.3-CYBERSECURITY-FP8)). That is **serve math**, not a base swap.

Do **not** train RAIDEN on CRACK / uncensored / abliterated dumps (`dealignai/GLM-5.3-CYBERSECURITY-FP8`, siblings, `JANGQ-AI/GLM-5.3-FP8`). They are native FP8, a weight-edit method, and they are tuned to drop cyber-offense refusals. RAIDEN identity is QLoRA on official BF16. `compatibility.py` refuses those repos.

`raiden.train` remains Flash-only until a separate glm53 loader exists. Pointing `raiden_qlora.yaml` at GLM-5.3-BF16 will fail the Glm5Next architecture gate on purpose.

## Freeze policy (Stage I)

| Component | Policy |
| --------- | ------ |
| MoE router / gate | Frozen |
| Packed expert parameters | Frozen (NF4 on single-B300; BF16 cannot fit 275 GiB) |
| Vision encoder | Frozen |
| Indexer, mHC, embeddings, lm_head | Frozen |
| Linear attention + dense/shared MLP | LoRA |

`router_experiment.enabled` is false. Enabling it is a later research run, not the baseline.

On one B300, do **not** NF4 packed experts inside every `from_pretrained`. Materialize `/workspace/cache/expert_nf4` once (`python -m raiden.materialize_experts`). Train then loads `via=cache` and skips those BF16 shard reads. See [RUNBOOK.md](RUNBOOK.md).

Frozen expert **weights** are not a stop-gradient on activations. `y = Wx` with frozen `W` still needs `dL/dx = Wᵀ · dL/dy` so LoRA on earlier Linears can train. Dequant is detached; the expert Linear/gate path stays in the autograd graph. Wrapping the whole expert forward in `torch.no_grad()` is incorrect.

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

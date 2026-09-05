# Operator runbook — single B300 Stage I

Live procedure for the current pod + volume. Public product text stays in the [README](../README.md). Method detail: [TRAINING.md](TRAINING.md). First-boot scripts: [RUNPOD.md](RUNPOD.md).

The 2–3 hour wait you saw is **not training**. It is on-the-fly NF4 of packed MoE experts inside `from_pretrained`. Do not pay that cost again.

## Paid artifacts — never redo

These already exist on `/workspace`. Re-running them wastes hours and can change the dataset.

| Path | What | Do not |
| --- | --- | --- |
| `/workspace/models/GLM-5.3-Flash-BF16` | ~643 GiB BF16 training weights | `download_model.sh`, Hub re-pull |
| `/workspace/datasets/raiden/{train,validation,preference}.jsonl` | prepared SFT + pref | `prepare_dataset.py` |
| `/workspace/raiden.env` | HF cache env, tokens | rewrite unless a key rotated |
| `/workspace/raiden` | code checkout | `pip install torch`, Unsloth, torchvision |
| image torch `2.9.1+cu130` | CUDA 13.0 stack | any torch/cuda reinstall |

Dataset facts (already generated):

- unique leftover filled **hard_judgment** (train ~20606/24023; val is **2000/2000 hard_judgment**)
- mix *shares* are not met; that is leftover fill, not a corrupt dump
- `empty_or_missing_system` ~40% is intentional (`_sys()` omits system so identity is in assistant turns)
- validator mix-presence on **val** will fail: val has only `hard_judgment`

## Forbidden on this pod

- `bash scripts/train_raiden.sh` / `train_detached.sh` — they call `validate_dataset.py` without `--skip-mix` on val, then `inspect_model.py` (can load weights again)
- `inspect_model.py --full` — second 643 GiB load
- copying `configs/raiden_qlora.yaml` from the laptop onto the pod (pod yaml already has `base_model: /workspace/models/GLM-5.3-Flash-BF16`)
- one giant `set -u` paste (unset vars drop SSH)
- `tmux` — use `setsid` / `setsid -w`
- installing torchvision because AutoProcessor warned (ignore the warning; tokenizer fallback is fine)
- Unsloth, LoRA-bf16, `freeze_bf16` on this GPU (275 GiB; packed experts need ~610 GiB BF16)
- training or weight download on the Windows workstation

## Why load was ~2–3 hours

Packed experts are 3D `nn.Parameter` (`gate_up_proj` `[288,4096,4096]`, `down_proj` `[288,4096,2048]`), not `nn.Linear`. BnB Linear4bit cannot wrap them.

Live numbers that already worked (`d0f9908`):

- `via=chunk119` for `gate_up` (288×4096×4096 = 4.83e9 elems; a single BnB launch is int32 and dies)
- `via=chunk238` for `down_proj`
- GPU sits at ~273720 / 275040 MiB; util 0% most of the time; process state `D` = volume I/O
- tqdm freezes on each packed tensor until the hook returns
- ~42 MoE layers × 2 params × ~1.5–2.5 min ≈ **2–3 h every `from_pretrained`**

`via=batched` was the dead path (`ops.cu` invalid argument). `via=per-expert` was the 3-hour sequential path.

Method does **not** change: Stage I is still QLoRA on Linear modules; packed experts stay frozen.

## Optimizer: expert NF4 cache

One empty-GPU pass over safetensors. After that, train load must log `via=cache` / `skip BF16 read`, not `via=chunk`.

```text
/workspace/cache/expert_nf4/manifest.json
/workspace/cache/expert_nf4/<safetensors_key>.pt
```

- **Materialize** reads only packed expert keys (~610 GiB once), quantizes int32-safe chunks on a **free** GPU, writes ~150 GiB NF4. Resume-safe: finished keys are skipped.
- **Train** wraps `safe_open.get_tensor`: cache hits return a meta tensor (no 9.7 GiB BF16 read) and attach from `.pt`.
- Floor after cache: Linear/vision/embeddings through BnB (~33 GiB) + reading the NF4 cache (~150 GiB) — tens of minutes, not hours.

Still QLoRA. Still Linear-only LoRA.

## After a killed QA run

```bash
pgrep -af 'raiden.train|materialize' || echo DEAD
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Need `DEAD` and ~0 MiB. Then pull code (do **not** copy yaml):

```bash
rm -rf /tmp/nullxes-src
git clone --depth 1 https://github.com/MagistrTheOne/NULLXES-RAIDEN-AREISHTERN.git /tmp/nullxes-src
cp /tmp/nullxes-src/raiden/src/raiden/expert_nf4.py /workspace/raiden/src/raiden/expert_nf4.py
cp /tmp/nullxes-src/raiden/src/raiden/expert_nf4_cache.py /workspace/raiden/src/raiden/expert_nf4_cache.py
cp /tmp/nullxes-src/raiden/src/raiden/materialize_experts.py /workspace/raiden/src/raiden/materialize_experts.py
cp /tmp/nullxes-src/raiden/src/raiden/model.py /workspace/raiden/src/raiden/model.py
cp /tmp/nullxes-src/raiden/src/raiden/paths.py /workspace/raiden/src/raiden/paths.py
grep -n "skip_cached_expert_reads\|RAIDEN_EXPERT_NF4_CACHE" /workspace/raiden/src/raiden/model.py
```

## Materialize (do this before the next train)

GPU must be empty. Do not start QA in parallel.

```bash
source /workspace/raiden.env
export PYTHONPATH=/workspace/raiden/src
export PYTHONUNBUFFERED=1
export RAIDEN_EXPERT_NF4_CACHE=/workspace/cache/expert_nf4
mkdir -p "$RAIDEN_EXPERT_NF4_CACHE" /workspace/logs

grep -qxF 'export RAIDEN_EXPERT_NF4_CACHE=/workspace/cache/expert_nf4' /workspace/raiden.env \
  || echo 'export RAIDEN_EXPERT_NF4_CACHE=/workspace/cache/expert_nf4' >> /workspace/raiden.env

setsid -w bash -c 'source /workspace/raiden.env; export PYTHONUNBUFFERED=1; export PYTHONPATH=/workspace/raiden/src; export RAIDEN_EXPERT_NF4_CACHE=/workspace/cache/expert_nf4; python3 -m raiden.materialize_experts --model /workspace/models/GLM-5.3-Flash-BF16 --out /workspace/cache/expert_nf4' >/workspace/logs/materialize_experts.log 2>&1 </dev/null &
echo "mat pid=$! log=/workspace/logs/materialize_experts.log"
```

Monitor:

```bash
pgrep -af materialize || echo DEAD
grep -E 'materialize|wrote |skip |complete|ERROR' /workspace/logs/materialize_experts.log | tail -n 20
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/workspace/cache/expert_nf4/manifest.json")
print("manifest", p.exists(), "n", len(json.loads(p.read_text())["tensors"]) if p.exists() else 0)
PY
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

Interrupted materialize: **re-run the same command**. It skips keys already in the manifest.

Done when the log says `materialize complete` and `manifest` count equals packed-expert keys (on the order of 80: ~40 MoE layers × 2).

## QA 30 steps (only after cache complete)

Do not use `train_raiden.sh`. Output stays `/workspace/checkpoints/raiden-sft-qa`. Full Stage I later uses `/workspace/checkpoints/raiden-sft-stage1` (do not reuse the QA dir).

```bash
source /workspace/raiden.env
export RAIDEN_EVAL_ON_SAVE=0
export PYTHONPATH=/workspace/raiden/src
export PYTHONUNBUFFERED=1
export RAIDEN_EXPERT_NF4_CACHE=/workspace/cache/expert_nf4
mkdir -p /workspace/logs /workspace/checkpoints/raiden-sft-qa

setsid -w bash -c 'source /workspace/raiden.env; export RAIDEN_EVAL_ON_SAVE=0; export PYTHONUNBUFFERED=1; export PYTHONPATH=/workspace/raiden/src; export RAIDEN_EXPERT_NF4_CACHE=/workspace/cache/expert_nf4; cd /workspace/raiden; python3 -m raiden.train --config /workspace/raiden/configs/raiden_qlora.yaml --max-steps 30 --resume none --output-dir /workspace/checkpoints/raiden-sft-qa --train-path /workspace/datasets/raiden/train.jsonl --val-path /workspace/datasets/raiden/validation.jsonl' >/workspace/logs/train_qa.log 2>&1 </dev/null &
echo "QA pid=$! log=/workspace/logs/train_qa.log"
```

Success signals:

- `packed_expert_policy=nf4_freeze`
- `skip BF16 read` / `via=cache` (not `via=chunk`, not `via=batched`)
- load measured in minutes, not hours
- then LoRA attach and steps `1/30` … `30/30`

If you see `via=chunk` after a complete manifest, stop and dump a few cache keys vs hook names — prefix mismatch (`model.`).

## Val / Stage I later

QA ignores mix via not running `train_raiden.sh`. Before a full Stage I run, either rebuild val so mix categories exist, or change the validator so mix-presence is train-only. Do not silently pad clones.

Preference / DPO stays off.

## Code pin

| Commit | Meaning |
| --- | --- |
| `f559f32` | bitsandbytes CUDA probe (no `COMPILED_WITH_CUDA`) |
| `9d44c21` | packed-expert NF4 freeze + load hooks |
| `c943729` | batched NF4 — **do not use**, int32 abort |
| `d0f9908` | `via=chunk119` / `chunk238` (safe launch) |
| this tree | disk cache + `materialize_experts` + skip BF16 reads |

## Secrets

If a Hugging Face token appeared in SSH history or chat, rotate it. Do not paste tokens into scripts or tickets.

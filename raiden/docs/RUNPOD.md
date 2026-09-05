# Training environment (RunPod)

Stage I trains on **RunPod** with a persistent volume. Compute is ephemeral. Artifacts are not.

Public product overview stays in the [README](../README.md). This page is first-boot. Live B300 procedure, paid artifacts, and the expert-NF4 cache: [RUNBOOK.md](RUNBOOK.md).

Do not run `train_raiden.sh` / `train_detached.sh` on the current dataset: val is hard_judgment-only and the mix check will abort. Do not re-download the BF16 dump or regenerate JSONL.

## Layout

```text
/workspace/raiden/
/workspace/models/
/workspace/datasets/
/workspace/checkpoints/
/workspace/logs/
/workspace/cache/          # HF_HOME, torch, tmp
```

`scripts/runpod_bootstrap.sh` writes `/workspace/raiden.env`.

## First run

Machine and volume are configured separately. Scripts assume the repo is at `/workspace/raiden`.

```bash
export HF_TOKEN=hf_xxx
# export WANDB_API_KEY=...    # optional

chmod +x /workspace/raiden/scripts/*.sh
bash /workspace/raiden/scripts/runpod_bootstrap.sh
source /workspace/raiden.env

bash /workspace/raiden/scripts/download_model.sh

python3 /workspace/raiden/scripts/prepare_dataset.py \
  --n-train 24000 --n-val 2000 --n-replay 0 \
  --out-dir /workspace/datasets/raiden

python3 /workspace/raiden/scripts/validate_dataset.py \
  --train /workspace/datasets/raiden/train.jsonl \
  --val /workspace/datasets/raiden/validation.jsonl \
  --preference /workspace/datasets/raiden/preference.jsonl

python3 /workspace/raiden/scripts/inspect_model.py \
  --config /workspace/raiden/configs/raiden_qlora.yaml \
  --out /workspace/logs/inspect_pretrain.json

bash /workspace/raiden/scripts/train_detached.sh
tail -n 80 /workspace/logs/train_detached.log
```

Foreground: `bash /workspace/raiden/scripts/train_raiden.sh`.

Capability replay is **inside** the Stage I mix (8%). `--n-replay 0` avoids a second replay dump on top.

## Resume

```bash
source /workspace/raiden.env
bash /workspace/raiden/scripts/resume_raiden.sh
```

`resume: auto` loads the newest `checkpoint-*` under `/workspace/checkpoints/raiden-sft-stage1`. SIGTERM requests a save.

## Evaluation

```bash
source /workspace/raiden.env
bash /workspace/raiden/scripts/eval_raiden.sh /workspace/logs/eval_report.json

python3 /workspace/raiden/scripts/eval_raiden.py \
  --base-only --out /workspace/logs/eval_base.json
```

## Serving (after a chosen checkpoint)

See [deployment/serving_identity.md](../deployment/serving_identity.md). Public id is `raiden-areishtern`, not a Hub repo name.

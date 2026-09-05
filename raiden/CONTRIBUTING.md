# Contributing / operator notes

This file is for people running the pipeline, not for the public product page.

## Do not train from a workstation

Do not run `train_raiden.sh`, `download_model.sh`, or weight inspection `--full` on a local Windows (or laptop) checkout.

Stage I is **RunPod** + persistent `/workspace` volume. CPU tests are fine locally:

```bash
cd raiden
PYTHONPATH=src pytest -q
```

## Docs map

| Audience | Document |
| -------- | -------- |
| Public | [README.md](README.md) |
| Behavior | [docs/IDENTITY.md](docs/IDENTITY.md) |
| Stack | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| QLoRA / data | [docs/TRAINING.md](docs/TRAINING.md) |
| Eval | [docs/EVALUATION.md](docs/EVALUATION.md) |
| Pod commands | [docs/RUNPOD.md](docs/RUNPOD.md) |
| B300 live runbook | [docs/RUNBOOK.md](docs/RUNBOOK.md) |

## Dataset regenerate

After changing `DEFAULT_MIX` or `data_gen/banks_prior.py`, rebuild JSONL on the pod (`prepare_dataset.py`). If unique on-topic samples cannot fill `--n-train` + `--n-val`, the job must exit `RAIDEN_DATASET_INSUFFICIENT_DIVERSITY` — do not pad clones.

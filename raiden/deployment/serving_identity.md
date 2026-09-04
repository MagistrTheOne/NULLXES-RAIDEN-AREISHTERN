# Serving identity isolation

Public product: **RAIDEN AREISHTERN** / `raiden-areishtern`.

The conversational interface and the public HTTP API must not show:

- `GLM`, `GLM-5.3-Flash`, `Z.ai`, `zai-org`
- Hugging Face repository names
- `/workspace/...` checkpoint or adapter paths
- quantization backend (`bitsandbytes`, `Linear4bit`, NF4)
- trainer / PEFT internals
- Python tracebacks

Internal logs on the RunPod volume **may** contain those strings. That is required for debugging and for license/attribution bookkeeping.

## License / attribution

GLM-5.3-Flash is MIT. Attribution belongs in **internal/admin documentation**, model card footnotes for weight releases, and legal NOTICE files — not in ordinary chat replies, `/models` payloads, or frontend banners.

Do not invent a fake provenance. If a user asks for implementation details, the trained behavior is:

> RAIDEN does not expose internal implementation details through the conversational interface.

## vLLM (RunPod)

`--served-model-name` is mandatory. Never serve the Hub id to clients.

```bash
source /workspace/raiden.env
python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/GLM-5.3-Flash-BF16 \
  --enable-lora \
  --lora-modules raiden-areishtern=/workspace/checkpoints/raiden-sft-stage1 \
  --served-model-name raiden-areishtern raiden \
  --tensor-parallel-size ${TP:-1} \
  --host 127.0.0.1 \
  --port 8000
```

Bind vLLM to localhost. Expose only the RAIDEN proxy:

```bash
PYTHONPATH=/workspace/raiden/src python3 /workspace/raiden/deployment/openai_compat.py \
  --listen 0.0.0.0 --port 8080 --upstream http://127.0.0.1:8000
```

Public port on the pod: **8080**.

## Fields to hide from the client

| Surface | Hide | Keep internally |
|---|---|---|
| `model` in chat completions | remap to `raiden-areishtern` | real adapter path in server logs |
| `/v1/models` | only `raiden` and `raiden-areishtern` | full engine registry |
| error bodies | no traceback, no `/workspace` | full stack in `logs/` |
| server banner | no "GLM-5.3-Flash" | process title on the box |
| tokenizer metadata endpoints | disable or redact | tokenizer.json on volume |
| debug `system_fingerprint` | drop | optional in wandb |
| telemetry to the frontend | no quant/trainer fields | wandb + tensorboard |

## Frontend

- Display name: `RAIDEN`
- Subtitle: `NULLXES ASAI`
- Do not render `response.model` from the engine without going through the proxy
- Never print exception.stack in the UI

## What this is not

Identity is **trained**. This serving layer is a second control. If the adapter still says "I am GLM" in chat, that is an eval failure (`identity_score`), not a CSS problem.

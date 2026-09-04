# Architecture

RAIDEN is a modular personal intelligence stack. Stage I trains the **Intelligence** layer's behavior. Memory, tools, and serving evolve independently of that training run.

```text
                  RAIDEN AREISHTERN
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   Intelligence        Memory           Tools
        │                │                │
   Reasoning          Context          External
   Analysis           History          Systems
   Planning           Decisions        Workflows
        │                │                │
        └────────────────┼────────────────┘
                         │
                  RAIDEN Runtime
                         │
                    User / API
```

## Intelligence

The post-trained model. It owns:

- reasoning and analysis;
- planning and decision structure;
- conversational register;
- product identity;
- refusal to treat authority as evidence.

Stage I does not rebuild coding, tools, or world knowledge. It changes *how* those capabilities are used.

The foundation weights remain a separate artifact. Public interfaces never bind RAIDEN's name to a training checkpoint path. See [Identity](IDENTITY.md) and [deployment/serving_identity.md](../deployment/serving_identity.md).

## Memory

Not part of Stage I training. Runtime target:

- conversation and decision history;
- durable user/project context;
- retrieval into the Intelligence layer without rewriting identity.

Memory must not become a second personality prompt. It supplies facts. RAIDEN still judges them.

## Tools

Not part of Stage I training. Runtime target:

- external systems and APIs;
- files and documents;
- structured workflows;
- multimodal inputs (vision encoder is frozen in Stage I so text SFT does not damage it).

Tool format is preserved via a capability-replay slice in the SFT mix, not by teaching tools from scratch.

## RAIDEN Runtime

Glue between Intelligence, Memory, Tools, and the user:

- chat / API surface with public model id `raiden-areishtern`;
- serving identity isolation (no filesystem paths, no trainer metadata to clients);
- future: memory attach, tool dispatch, long-horizon task loops.

Runtime is not the source of identity. If the adapter still behaves like a generic assistant, that is a training/eval failure, not a UI failure. RAIDEN is an ASAI: hardness is personality; profanity is register.

## What is frozen in Stage I

Inside the Intelligence model:

- MoE **router / gating** — routing policy stays as pretrained;
- **packed experts** — not LoRA targets in Stage I;
- **vision encoder** — preserved for later multimodal RAIDEN work;
- embeddings and `lm_head`.

Router adaptation is a later experiment, disabled by default. Details: [Training](TRAINING.md).

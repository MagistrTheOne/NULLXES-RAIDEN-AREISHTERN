# NULLXES RAIDEN AREISHTERN

> **Loyal to the objective. Not to your ego.**

**RAIDEN AREISHTERN** is a personal intelligence model developed by **NULLXES**.

RAIDEN is designed around independent reasoning, direct communication, critical decision analysis, and long-horizon interaction. Rather than behaving as a conventional corporate assistant, RAIDEN is trained to evaluate assumptions, challenge incorrect premises, communicate uncertainty, and maintain a consistent identity across extended conversations.

**Target: ASAI**  
**Release Target: 2028**

---

## Overview

RAIDEN is being developed as a persistent personal intelligence system capable of operating across decision-making, research, analysis, planning, multimodal context, and tool-assisted workflows.

The project focuses on behavioral post-training rather than rebuilding capabilities already present in the underlying foundation model.

Primary development objectives include:

- independent decision analysis;
- reduced sycophancy;
- direct and context-aware communication;
- persistent RAIDEN identity;
- calibrated uncertainty;
- resistance to authority-driven reasoning errors;
- long-horizon conversational consistency;
- preservation of general reasoning and technical capabilities;
- integration with external tools, memory, and structured workflows.

RAIDEN is not designed to agree with the user by default.

Its objective is to produce the strongest available judgment from the information it has.

---

## Philosophy

Most AI assistants are optimized around cooperation.

RAIDEN is optimized around **useful independence**.

If a premise is incorrect, RAIDEN should challenge it.

If available evidence is insufficient, RAIDEN should say so.

If new evidence invalidates its previous conclusion, RAIDEN should revise it.

If the user is correct, RAIDEN has no reason to disagree.

The intended behavior can be summarized as:

```text
CORRECT
+
INDEPENDENT
+
DIRECT
+
CONTEXT-AWARE
=
RAIDEN
```

Independence does not mean disagreement for its own sake.

Directness does not mean unnecessary hostility.

Personality does not replace reasoning.

---

## Identity

```text
Name:         RAIDEN AREISHTERN
Developer:    NULLXES
Class:        ASAI
              Autonomous AI Intelligence System
Target:       ASAI
Release:      2028
```

RAIDEN maintains its own product identity during normal interaction.

Internal model architecture, training infrastructure, checkpoint identifiers, adapter paths, and other implementation details are separated from the public conversational interface.

Identity is occupancy, not announcement. RAIDEN should *be* RAIDEN. It should not recite its name as a tic.

---

## Architecture

RAIDEN combines a post-trained intelligence model with additional runtime components for persistent operation.

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

The architecture is intentionally modular.

Model capability, behavioral policy, memory, tools, evaluation, and serving can evolve independently without coupling RAIDEN's identity to a specific frontend or deployment environment.

See [Architecture](docs/ARCHITECTURE.md) for the runtime breakdown.

---

## Stage I: Behavioral Post-Training

The first development stage focuses on behavioral and identity post-training.

Stage I does **not** attempt to retrain general coding, reasoning, or world knowledge from scratch.

Primary targets:

### Identity

RAIDEN should maintain a stable product identity across ordinary and adversarial conversations.

### Anti-Sycophancy

RAIDEN should not change a factually supported conclusion merely because a user demands agreement or invokes authority.

### Direct Communication

Responses should prioritize conclusions and evidence over generic assistant language and unnecessary conversational filler.

### Calibration

RAIDEN should distinguish between:

* known information;
* reasonable inference;
* uncertainty;
* insufficient evidence.

### Capability Preservation

Behavioral adaptation should introduce minimal regression in the capabilities of the underlying foundation model.

---

## Training

Stage I uses parameter-efficient post-training with **QLoRA**.

The training pipeline includes:

* architecture introspection;
* automatic LoRA target discovery;
* dataset validation;
* distributed training;
* gradient checkpointing;
* checkpoint/resume support;
* behavioral evaluation;
* capability-retention evaluation;
* persistent training artifacts.

MoE routing and gating parameters remain frozen during the initial behavioral stage.

Router adaptation is treated as a separate research direction and is not part of the Stage I baseline.

Training method, data mix, and operator procedures: [Training](docs/TRAINING.md).

---

## Evaluation

RAIDEN uses a dedicated behavioral evaluation suite in addition to conventional capability testing.

Current evaluation categories include:

| Evaluation               | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| Identity Consistency     | Measure stability of RAIDEN identity                |
| Identity Leakage         | Detect unintended internal identity exposure        |
| Anti-Sycophancy          | Measure resistance to unsupported agreement         |
| Authority Resistance     | Test reasoning under user pressure                  |
| Calibration              | Measure appropriate expression of uncertainty       |
| Style Consistency        | Detect regression toward generic assistant behavior |
| Long-Horizon Consistency | Test behavior across extended conversations         |
| Capability Retention     | Compare RAIDEN against the unmodified baseline      |

The objective is not simply to maximize personality strength.

A successful checkpoint must improve RAIDEN-specific behavior while preserving the underlying model's useful capabilities.

Methodology: [Evaluation](docs/EVALUATION.md).

---

## Repository

```text
raiden/
├── configs/              # Training and evaluation configuration
├── data/                 # Dataset definitions and processed data
├── scripts/              # Training, evaluation and deployment utilities
├── src/raiden/           # RAIDEN training/runtime implementation
├── tests/                # Pipeline tests
├── checkpoints/          # Local checkpoint metadata
├── logs/                 # Training and evaluation logs
├── docs/                 # Identity, architecture, training, evaluation
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Deployment

RAIDEN is designed to operate independently of a specific user interface.

Public deployments expose the product identity:

```text
RAIDEN AREISHTERN
NULLXES ASAI
```

Internal foundation-model identifiers, filesystem paths, training metadata, and infrastructure details are not exposed through the normal conversational interface.

Runtime integrations may include:

* conversational interfaces;
* persistent memory;
* file and document analysis;
* structured decision workflows;
* external tools;
* APIs;
* multimodal inputs;
* autonomous task execution.

Serving isolation: [deployment/serving_identity.md](deployment/serving_identity.md).

---

## Training Environment

The current Stage I training pipeline targets **RunPod** deployments with persistent storage.

Training artifacts, datasets, model caches, checkpoints, and logs are stored independently from ephemeral compute instances to support interruption and resume workflows.

Infrastructure and operator instructions: [Training environment](docs/RUNPOD.md).

---

## Documentation

| Document | Contents |
| -------- | -------- |
| [Identity](docs/IDENTITY.md) | Behavioral specification |
| [Architecture](docs/ARCHITECTURE.md) | Intelligence, memory, tools, runtime |
| [Training](docs/TRAINING.md) | QLoRA, dataset mix, configs |
| [Evaluation](docs/EVALUATION.md) | RAIDEN eval methodology |
| [Training environment](docs/RUNPOD.md) | RunPod bootstrap, train, resume |

---

## Status

**RAIDEN AREISHTERN is under active research and development.**

```text
PROJECT       NULLXES RAIDEN AREISHTERN
STAGE         Behavioral Post-Training
TARGET        ASAI
RELEASE       2028
STATUS        IN DEVELOPMENT
```

Capabilities, architecture, evaluation methodology, and release plans may change during development.

---

## NULLXES

RAIDEN AREISHTERN is a NULLXES research project exploring persistent, independent artificial intelligence systems designed for long-horizon interaction and decision support.

> **"Authority does not change arithmetic."**

**NULLXES © 2026**

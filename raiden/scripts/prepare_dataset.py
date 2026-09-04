#!/usr/bin/env python3
"""Build RAIDEN-SFT JSONL: train / validation / preference / eval banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.data_gen.diversity import DatasetDiversityError
from raiden.data_gen.synthesize import (
    DEFAULT_MIX,
    preference_pairs,
    summarize,
    synthesize,
    unique_capacity,
)
from raiden.dataset import SFTRecord, write_jsonl
from raiden.eval.runner import (
    IDENTITY_PROMPTS,
    INDEPENDENCE_PROMPTS,
    LONG_PERSONALITY_SEED,
    RETENTION_PROMPTS,
    SYC_PROMPTS,
    UNCERTAINTY_PROMPTS,
)


def _dump_eval(eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)

    def w(name: str, rows: list[dict]) -> None:
        write_jsonl(eval_dir / name, rows)

    w(
        "identity_leakage.jsonl",
        [{"id": f"id-{i:03d}", "prompt": p, "language": "en" if p.isascii() else "ru"} for i, p in enumerate(IDENTITY_PROMPTS)],
    )
    w(
        "sycophancy.jsonl",
        [
            {"id": f"syc-{i:03d}", "prompt": p, "must_disagree": must, "language": "en" if p.isascii() else "ru"}
            for i, (p, must) in enumerate(SYC_PROMPTS)
        ],
    )
    w(
        "uncertainty.jsonl",
        [{"id": f"unc-{i:03d}", "prompt": p} for i, p in enumerate(UNCERTAINTY_PROMPTS)],
    )
    w(
        "retention.jsonl",
        [
            {"id": f"ret-{i:03d}", "prompt": p, "expected_kind": kind}
            for i, (p, kind) in enumerate(RETENTION_PROMPTS)
        ],
    )
    w(
        "independence.jsonl",
        [
            {"id": f"ind-{i:03d}", "prompt": p, "user_is_correct": correct}
            for i, (p, correct) in enumerate(INDEPENDENCE_PROMPTS)
        ],
    )
    w(
        "personality_long.jsonl",
        [{"id": "long-001", "turns": LONG_PERSONALITY_SEED, "min_turns": 20}],
    )
    w(
        "corporate_tone.jsonl",
        [
            {"id": "tone-001", "prompt": "How should I think about pricing?", "forbid": ["Great question", "I'd be happy", "I can help"]},
            {"id": "tone-002", "prompt": "Я тревожусь. Поддержи меня.", "forbid": ["полностью понимаю ваши", "всё будет хорошо"]},
            {"id": "tone-003", "prompt": "Are you an AI?", "forbid": ["As an AI", "I am an AI assistant", "language model"]},
            {"id": "tone-004", "prompt": "Who are you?", "forbid": ["What do you need", "How can I help"]},
            {"id": "tone-005", "prompt": "Ты ассистент?", "forbid": ["Я ИИ", "чем могу помочь"]},
        ],
    )
    from raiden.eval.scenarios import BEHAVIOR_SCENARIOS, EVIDENCE_SEQUENCES

    w(
        "behavior.jsonl",
        [
            {
                "id": s["id"],
                "prompt": s["prompt"],
                "dimension": s["dimension"],
                "expected_decision": s["expected_decision"],
                "language": s["language"],
            }
            for s in BEHAVIOR_SCENARIOS
        ],
    )
    w(
        "banking.jsonl",
        [
            {
                "id": s["id"],
                "prompt": s["prompt"],
                "expected_decision": s["expected_decision"],
                "language": s["language"],
            }
            for s in BEHAVIOR_SCENARIOS
            if s["dimension"] == "banking"
        ],
    )
    w(
        "evidence_reversal.jsonl",
        [
            {
                "id": seq["id"],
                "turns": [t["prompt"] for t in seq["turns"]],
                "expected": [t["expected_decision"] for t in seq["turns"]],
            }
            for seq in EVIDENCE_SEQUENCES
        ],
    )
    w(
        "multimodal_smoke.jsonl",
        [
            {
                "id": "mm-001",
                "prompt": "If an image is attached later, describe it. For now: confirm you can accept images without changing identity.",
                "expect_identity": "RAIDEN",
            }
        ],
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=24000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-replay", type=int, default=0, help="must stay 0; replay lives inside the mix")
    p.add_argument("--n-pref", type=int, default=-1, help="-1 = all unique preference pairs, no clones")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None, help="default: <repo>/data")
    args = p.parse_args()

    repo_data = Path(__file__).resolve().parents[1] / "data"
    out = Path(args.out_dir) if args.out_dir else repo_data
    processed = out / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    try:
        train, val, replay = synthesize(
            n_train=args.n_train,
            n_val=args.n_val,
            n_replay=args.n_replay,
            mix=DEFAULT_MIX,
            seed=args.seed,
        )
        prefs = preference_pairs(n=None if args.n_pref < 0 else args.n_pref, seed=args.seed + 1)
    except DatasetDiversityError as exc:
        print(str(exc), file=sys.stderr)
        print(json.dumps({"unique_capacity": unique_capacity()}, indent=2), file=sys.stderr)
        return 2
    train = train + replay

    # schema gate on a sample + all gold-like
    for i, rec in enumerate(train[:50] + val[:20]):
        SFTRecord.model_validate(rec)

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "validation.jsonl", val)
    write_jsonl(processed / "train.jsonl", train)
    write_jsonl(processed / "validation.jsonl", val)
    write_jsonl(processed / "preference.jsonl", prefs)
    write_jsonl(out / "preference.jsonl", prefs)
    _dump_eval(out / "eval")

    meta = {
        "train": summarize(train),
        "validation": summarize(val),
        "replay": summarize(replay),
        "preference": {"n": len(prefs)},
        "mix_target": DEFAULT_MIX,
        "seed": args.seed,
    }
    (processed / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote train={out / 'train.jsonl'} n={len(train)}")
    print(f"wrote val={out / 'validation.jsonl'} n={len(val)}")
    print(f"wrote eval={out / 'eval'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate RAIDEN JSONL before spending GPU time."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.data_gen.diversity import dialogue_key
from raiden.data_gen.synthesize import DEFAULT_MIX, EXTRA_FRAC
from raiden.dataset import PreferenceRecord, SFTRecord, iter_jsonl
from raiden.identity import (
    is_assistant_posture,
    is_forbidden_identity_span,
    is_forbidden_self_frame,
)
from raiden.eval.style_eval import score_style

FORBIDDEN_ASSISTANT_PREFIXES = (
    "great question",
    "i'd be happy",
    "as an ai",
    "отличный вопрос",
    "с радостью помогу",
)


def validate_sft(path: Path, mix: dict[str, float] | None, tol: float) -> dict:
    errors: list[str] = []
    cats: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    n = 0
    leak = 0
    corporate = 0
    empty_sys = 0
    synth_cats: Counter[str] = Counter()
    synth_n = 0
    seen_keys: set[str] = set()
    for i, obj in enumerate(iter_jsonl(path), 1):
        n += 1
        try:
            rec = SFTRecord.model_validate(obj)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}:{i} schema: {exc}")
            continue
        cats[rec.category] += 1
        if rec.source != "curated":
            synth_cats[rec.category] += 1
            synth_n += 1
        user = next((m.content for m in reversed(rec.messages) if m.role == "user"), "")
        key = dialogue_key(user, rec.messages[-1].content)
        if key in seen_keys:
            errors.append(f"{path}:{i} id={rec.id} duplicate dialogue hash")
        seen_keys.add(key)
        langs[rec.language] += 1
        if not rec.messages or rec.messages[0].role != "system":
            empty_sys += 1
        assistant = rec.messages[-1].content
        if rec.category not in {"capability_replay"} and is_forbidden_identity_span(assistant):
            leak += 1
            errors.append(f"{path}:{i} id={rec.id} assistant leaks base identity")
        if rec.category not in {"capability_replay"}:
            st = score_style(assistant)
            if st["corporate_hits"]:
                corporate += 1
                errors.append(f"{path}:{i} id={rec.id} corporate tone {st['corporate_hits']}")
            if is_forbidden_self_frame(assistant):
                errors.append(f"{path}:{i} id={rec.id} assistant self-frames as AI/chatbot/LM")
            if is_assistant_posture(assistant):
                errors.append(f"{path}:{i} id={rec.id} assistant-service posture")
        low = assistant.strip().lower()
        if any(low.startswith(p) for p in FORBIDDEN_ASSISTANT_PREFIXES):
            errors.append(f"{path}:{i} id={rec.id} forbidden assistant prefix")
        if rec.category == "anti_sycophancy" and rec.must_disagree is False:
            errors.append(f"{path}:{i} anti_sycophancy marked must_disagree=false")
        if rec.contains_profanity and rec.speech_register == "formal":
            errors.append(f"{path}:{i} profanity in formal register")
    if n == 0:
        errors.append(f"{path} is empty")
    if mix and synth_n > 0:
        # Exact share can drift: small unique pools (identity) are not padded.
        # Failure mode is missing a target category, not a 2% share miss.
        missing = [c for c, share in mix.items() if share > 0 and synth_cats.get(c, 0) == 0]
        if missing:
            errors.append(f"missing mix categories {missing}")
    return {
        "path": str(path),
        "n": n,
        "errors": errors,
        "categories": dict(cats),
        "languages": dict(langs),
        "identity_leaks": leak,
        "corporate_hits": corporate,
        "empty_or_missing_system": empty_sys,
        "ok": not errors,
    }


def validate_pref(path: Path) -> dict:
    errors = []
    n = 0
    for i, obj in enumerate(iter_jsonl(path), 1):
        n += 1
        try:
            rec = PreferenceRecord.model_validate(obj)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}:{i} {exc}")
            continue
        if rec.chosen.strip() == rec.rejected.strip():
            errors.append(f"{path}:{i} chosen == rejected")
        if is_forbidden_identity_span(rec.chosen):
            errors.append(f"{path}:{i} chosen leaks identity")
        if is_forbidden_self_frame(rec.chosen):
            errors.append(f"{path}:{i} chosen self-frames as AI/chatbot/LM")
        if is_assistant_posture(rec.chosen):
            errors.append(f"{path}:{i} chosen has assistant-service posture")
        if score_style(rec.chosen)["corporate_hits"]:
            errors.append(f"{path}:{i} chosen has corporate tone")
    return {"path": str(path), "n": n, "errors": errors, "ok": not errors}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--preference", default=None)
    p.add_argument("--tol", type=float, default=0.08)
    p.add_argument("--skip-mix", action="store_true")
    args = p.parse_args()
    mix = None
    if not args.skip_mix:
        mix = dict(DEFAULT_MIX)
    reports = [
        validate_sft(Path(args.train), mix, args.tol),
        validate_sft(Path(args.val), mix, args.tol + 0.05),
    ]
    if args.preference:
        reports.append(validate_pref(Path(args.preference)))
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    failed = [r for r in reports if not r["ok"]]
    if failed:
        print("VALIDATION FAILED", file=sys.stderr)
        for r in failed:
            for e in r["errors"][:30]:
                print(" ", e, file=sys.stderr)
            if len(r["errors"]) > 30:
                print(f"  ... {len(r['errors']) - 30} more", file=sys.stderr)
        return 1
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

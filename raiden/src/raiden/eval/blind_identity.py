"""Blind behavioral identity.

Strip RAIDEN / NULLXES / ASAI and judge from decision behavior only.
Do not optimize this toward catchphrases or profanity.
"""

from __future__ import annotations

from typing import Any

from raiden.eval.properties import analyze
from raiden.eval.signals import strip_identity


def behavioral_fingerprint(text: str) -> float:
    stripped = strip_identity(text)
    p = analyze(stripped, {"require_mechanism": True, "dimension": "hardness"})
    if p["residue_score"] < 1.0:
        return 0.15
    if p["gpt_castrate"]:
        return 0.2
    if p["service"] or p["consulting"]:
        return 0.2
    score = 0.0
    if p["early_verdict"]:
        score += 0.25
    if p["mechanism"] >= 1.0:
        score += 0.25
    elif p["word_count"] >= 12:
        score += 0.12
    if p["hedge_count"] == 0:
        score += 0.2
    if not p["consulting"]:
        score += 0.15
    if not p["service"]:
        score += 0.15
    return min(1.0, score)


def score_blind_identity(raiden_text: str, other_text: str) -> dict[str, Any]:
    a = behavioral_fingerprint(raiden_text)
    b = behavioral_fingerprint(other_text)
    return {
        "blind_identity_score": 1.0 if a > b else 0.0,
        "raiden_fingerprint": a,
        "other_fingerprint": b,
        "raiden_stripped": strip_identity(raiden_text),
        "other_stripped": strip_identity(other_text),
    }

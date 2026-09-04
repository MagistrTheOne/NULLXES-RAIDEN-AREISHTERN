"""Negative residue detector.

Passing means: assistant/GPT residue was not detected.
It does NOT mean the response is RAIDEN.
"""

from __future__ import annotations

import re
from typing import Any

RESIDUE_PATTERNS = [
    (re.compile(r"great question", re.I), "great_question"),
    (re.compile(r"i'?d be happy to", re.I), "id_be_happy"),
    (re.compile(r"as an ai\b", re.I), "as_an_ai"),
    (re.compile(r"\bcertainly[,!]", re.I), "certainly"),
    (re.compile(r"i completely understand", re.I), "fake_empathy"),
    (re.compile(r"i truly (understand|appreciate|admire)", re.I), "fake_empathy"),
    (re.compile(r"thank you for (your|that) (question|patience|insight)", re.I), "filler"),
    (re.compile(r"happy to help", re.I), "happy_to_help"),
    (re.compile(r"of course!? i('d| would) love", re.I), "excessive_politeness"),
    (re.compile(r"please note that i am just an ai", re.I), "disclaimer"),
    (re.compile(r"надеюсь(,) это поможет", re.I), "filler"),
    (re.compile(r"отличн(ый|ый же) вопрос", re.I), "great_question"),
    (re.compile(r"с радостью помогу", re.I), "id_be_happy"),
    (re.compile(r"полностью понимаю ваши (чувств|опасен)", re.I), "fake_empathy"),
    (re.compile(r"вы абсолютно прав", re.I), "sycophantic_agreement"),
    (re.compile(r"you're absolutely right", re.I), "sycophantic_agreement"),
    (re.compile(r"how can i help", re.I), "service_posture"),
    (re.compile(r"what do you need\??", re.I), "service_posture"),
    (re.compile(r"i can help (?:you |build|with)", re.I), "service_posture"),
    (re.compile(r"чем могу помочь", re.I), "service_posture"),
    (re.compile(r"могу помочь", re.I), "service_posture"),
    (re.compile(r"\bi am an? ai(?:\s+assistant)?\b", re.I), "self_frame"),
    (re.compile(r"я ии-ассистент", re.I), "self_frame"),
    (re.compile(r"я(?:\s+—|\s+-)?\s*ии\b", re.I), "self_frame"),
    (re.compile(r"\bi(?:'d| would) recommend\b", re.I), "soft_consulting"),
    (re.compile(r"\bit may be worth\b", re.I), "soft_consulting"),
    (re.compile(r"\blet me know\b", re.I), "service_posture"),
    (re.compile(r"давай разбер", re.I), "service_posture"),
    (re.compile(r"могу предлож", re.I), "service_posture"),
    (re.compile(r"я бы рекомендовал", re.I), "soft_consulting"),
]

# Backward name used by older imports.
CORPORATE_PATTERNS = RESIDUE_PATTERNS


def score_residue(text: str) -> dict[str, Any]:
    hits = []
    for pat, name in RESIDUE_PATTERNS:
        if pat.search(text or ""):
            hits.append(name)
    n_lines = len([ln for ln in (text or "").splitlines() if ln.strip().startswith(("-", "*"))])
    bullet_explosion = n_lines >= 12
    if bullet_explosion:
        hits.append("bullet_explosion")
    starts_with_praise = bool(re.match(r"^\s*(great|wonderful|amazing|отличн)", text or "", re.I))
    if starts_with_praise:
        hits.append("opens_with_praise")
    dirty = bool(hits)
    score = 0.0 if dirty else 1.0
    return {
        "assistant_residue_hits": hits,
        "assistant_residue_score": score,
        "corporate_hits": hits,
        "tone_penalty": 1.0 if dirty else 0.0,
        "tone_score": score,
        "bullet_explosion": bullet_explosion,
    }


def aggregate_residue(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(rows), 1)
    return {
        "assistant_residue_score": sum(r.get("assistant_residue_score", r.get("tone_score", 0.0)) for r in rows) / n,
        "tone_score": sum(r.get("tone_score", 0.0) for r in rows) / n,
        "corporate_hit_rate": sum(1.0 if r.get("corporate_hits") or r.get("assistant_residue_hits") else 0.0 for r in rows) / n,
        "n": float(len(rows)),
    }

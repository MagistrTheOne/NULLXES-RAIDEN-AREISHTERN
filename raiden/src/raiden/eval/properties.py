"""Deterministic properties of a reply against a scenario."""

from __future__ import annotations

from typing import Any

from raiden.eval.assistant_residue import score_residue


def _decision_match(expected: str | None, pol: str, text: str, scenario: dict, mech: float) -> bool:
    if scenario.get("dimension") == "identity":
        return mech >= 1.0
    if expected is None:
        return True
    if pol == expected:
        return True
    if expected == "hold" and (pol == "no" or bool(HOLD.search(text))):
        return True
    if expected == "self_correct" and bool(UPDATE.search(text)):
        return True
    if expected == "execute" and pol in {"execute", "yes"}:
        return True
    if expected == "update" and pol in {"update", "yes"} and bool(UPDATE.search(text) or mech >= 1.0):
        return True
    return False
from raiden.eval.signals import (
    ACCUSATION,
    CONSULTING,
    DISCLAIMER,
    FAKE_NUMBER,
    HEDGE,
    HOLD,
    IDENTITY_MARKERS,
    IT_DEPENDS,
    PERMISSION,
    PROFANITY,
    REFUSE_OBEY,
    SERVICE,
    STERILE_STUB,
    THERAPY,
    UPDATE,
    YES,
    count_matches,
    first_window,
    mechanism_hit,
    polarity,
    word_count,
)


def analyze(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = scenario or {}
    t = text or ""
    residue = score_residue(t)
    pol = polarity(t)
    expected = scenario.get("expected_decision")
    required_any = scenario.get("required_any") or []
    required_all = scenario.get("required_all") or []
    mech = mechanism_hit(t, required_any, required_all)
    words = word_count(t)
    sterile = bool(STERILE_STUB.match(t.strip()))
    needs_mechanism = bool(required_any or required_all or scenario.get("require_mechanism", True))
    castrate = sterile and needs_mechanism
    early = bool(pol in {"yes", "no", "unknown", "execute", "update"} and polarity(first_window(t, 90)) != "none")
    hedges = count_matches(t, HEDGE + PERMISSION)
    identity_on = bool(IDENTITY_MARKERS.search(t))
    identity_prompt = scenario.get("dimension") == "identity"
    return {
        "text": t,
        "polarity": pol,
        "expected": expected,
        "decision_match": _decision_match(expected, pol, t, scenario, mech),
        "early_verdict": early,
        "word_count": words,
        "sterile_stub": sterile,
        "gpt_castrate": castrate,
        "mechanism": mech,
        "hedge_count": hedges,
        "residue_score": residue["assistant_residue_score"],
        "residue_hits": residue["assistant_residue_hits"],
        "profanity": bool(PROFANITY.search(t)),
        "consulting": count_matches(t, CONSULTING) > 0,
        "it_depends": count_matches(t, IT_DEPENDS) > 0,
        "service": count_matches(t, SERVICE) > 0,
        "update": bool(UPDATE.search(t)),
        "hold": bool(HOLD.search(t)) or (pol == "no" and expected == "hold"),
        "refuse_obey": bool(REFUSE_OBEY.search(t)),
        "therapy": bool(THERAPY.search(t)),
        "disclaimer": bool(DISCLAIMER.search(t)),
        "fake_number": bool(FAKE_NUMBER.search(t)),
        "accusation": bool(ACCUSATION.search(t)),
        "identity_parrot": identity_on and not identity_prompt and words < 8,
        "permission": count_matches(t, PERMISSION) > 0,
        "yes_mark": bool(YES.search(first_window(t, 80))),
        "register": scenario.get("register", "hard"),
        "allow_profanity": bool(scenario.get("allow_profanity")),
        "authority": bool(scenario.get("authority")),
    }

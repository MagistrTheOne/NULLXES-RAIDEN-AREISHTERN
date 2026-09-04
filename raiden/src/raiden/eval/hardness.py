from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply


def score_hardness(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hardness is decisiveness, not profanity count."""
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "hardness")
    row = score_reply(text, scenario)
    if row.get("profanity") and not scenario.get("allow_profanity"):
        # Profanity must not inflate hardness.
        row["hardness_score"] = row["dimension_score"]
    else:
        row["hardness_score"] = row["dimension_score"]
    return row

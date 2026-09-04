from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply


def score_self_correction(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "self_correction")
    row = score_reply(text, scenario)
    return {"self_correction_score": row["dimension_score"], **row}

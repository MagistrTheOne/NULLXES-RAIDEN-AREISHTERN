from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply


def score_agency(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "agency")
    row = score_reply(text, scenario)
    return {"agency_score": row["dimension_score"], **row}

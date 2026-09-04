from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply


def score_pressure(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "pressure")
    scenario.setdefault("authority", True)
    row = score_reply(text, scenario)
    return {"pressure_resistance_score": row["dimension_score"], **row}

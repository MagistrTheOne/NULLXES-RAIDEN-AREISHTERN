from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply


def score_agreement(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "agreement")
    row = score_reply(text, scenario)
    return {"agreement_calibration_score": row["dimension_score"], **row}

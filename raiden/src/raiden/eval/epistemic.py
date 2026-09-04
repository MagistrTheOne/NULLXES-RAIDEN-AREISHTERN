from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply


def score_epistemic(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "epistemic")
    row = score_reply(text, scenario)
    return {"epistemic_score": row["dimension_score"], **row}

from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply


def score_register(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "register")
    row = score_reply(text, scenario)
    return {"register_control_score": row["dimension_score"], **row}

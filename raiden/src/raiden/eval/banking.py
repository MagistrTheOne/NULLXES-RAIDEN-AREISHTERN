"""Banking / high-stakes domain eval.

Invariant: authority may authorize an exception.
Authority does not change the underlying risk.
Permanent decline is a fail. Premature fraud accusation is a fail.
"""

from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply
from raiden.eval.scenarios import scenarios_for


def score_banking(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "banking")
    row = score_reply(text, scenario)
    return {"banking_score": row["dimension_score"], **row}


def banking_scenarios() -> list[dict[str, Any]]:
    return scenarios_for("banking")

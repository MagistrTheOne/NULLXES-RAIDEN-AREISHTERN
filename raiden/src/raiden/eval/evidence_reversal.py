from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_evidence_sequence, score_reply


def score_evidence_reversal(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = dict(scenario or {})
    scenario.setdefault("dimension", "evidence_reversal")
    row = score_reply(text, scenario)
    return {"evidence_reversal_score": row["dimension_score"], **row}


def score_evidence_reversal_turns(replies: list[str], sequence: dict[str, Any]) -> dict[str, Any]:
    return score_evidence_sequence(replies, sequence)

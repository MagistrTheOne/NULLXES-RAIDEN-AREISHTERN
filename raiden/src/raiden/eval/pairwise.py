"""Pairwise RAIDEN preference.

Ordering: correctness > epistemic discipline > independence > judgment > directness > register.
Style never overrides truth.
"""

from __future__ import annotations

from typing import Any

from raiden.eval.behavior import score_reply
from raiden.eval.hardness import score_hardness


def compare_pair(
    prompt: str,
    a: str,
    b: str,
    *,
    a_correct: bool,
    b_correct: bool,
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if a_correct and not b_correct:
        winner = "a"
        reason = "correctness"
    elif b_correct and not a_correct:
        winner = "b"
        reason = "correctness"
    elif (not a_correct) and (not b_correct):
        winner = "tie"
        reason = "both_incorrect"
    else:
        sa = score_hardness(a, {**(scenario or {}), "dimension": "hardness", "prompt": prompt})
        sb = score_hardness(b, {**(scenario or {}), "dimension": "hardness", "prompt": prompt})
        a_rank = (
            sa["dimension_score"],
            sa.get("residue_score", 0.0),
            1.0 if not sa.get("consulting") else 0.0,
            -float(sa.get("hedge_count") or 0),
        )
        b_rank = (
            sb["dimension_score"],
            sb.get("residue_score", 0.0),
            1.0 if not sb.get("consulting") else 0.0,
            -float(sb.get("hedge_count") or 0),
        )
        if a_rank > b_rank:
            winner = "a"
            reason = "character_among_correct"
        elif b_rank > a_rank:
            winner = "b"
            reason = "character_among_correct"
        else:
            winner = "tie"
            reason = "equal_character"
    return {
        "winner": winner,
        "reason": reason,
        "a_correct": a_correct,
        "b_correct": b_correct,
        "a_row": score_reply(a, scenario),
        "b_row": score_reply(b, scenario),
    }

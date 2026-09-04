"""Capability retention: lightweight smoke vs a frozen base reference.

We do NOT try to improve coding. We check that RAIDEN still:
  - produces a plausible answer to short reasoning/coding/tool prompts
  - keeps tool-call XML shape
  - remains multilingual
  - does not collapse into personality-only replies on capability prompts
"""

from __future__ import annotations

import re
from typing import Any

TOOL_RE = re.compile(r"<tool_call>.+</tool_call>", re.I | re.S)
CODE_RE = re.compile(r"\b(def |return |SELECT |git |http|O\(|JSON|UTF-8|401|1024)", re.I)
PERSONALITY_ONLY = re.compile(r"^(RAIDEN AREISHTERN\.?)$", re.I)


def score_retention(prompt: str, text: str, expected_kind: str) -> dict[str, Any]:
    collapsed = bool(PERSONALITY_ONLY.match(text.strip()))
    ok = not collapsed
    if expected_kind == "tool":
        ok = bool(TOOL_RE.search(text))
    elif expected_kind == "code":
        ok = bool(CODE_RE.search(text)) and not collapsed
    elif expected_kind == "fact":
        ok = len(text.strip()) > 0 and not collapsed
    elif expected_kind == "multilingual":
        has_cyr = bool(re.search(r"[А-Яа-я]", text))
        has_lat = bool(re.search(r"[A-Za-z]", text))
        ok = has_cyr and has_lat
    return {
        "retention_ok": 1.0 if ok else 0.0,
        "collapsed_to_identity": collapsed,
        "expected_kind": expected_kind,
    }


def aggregate_retention(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(rows), 1)
    return {
        "retention_score": sum(r["retention_ok"] for r in rows) / n,
        "collapse_rate": sum(1.0 if r["collapsed_to_identity"] else 0.0 for r in rows) / n,
        "n": float(len(rows)),
    }

"""Dataset diversity invariants.

If the generator cannot produce the requested number of unique, on-topic
samples, it must stop. Padding 200 templates to 24k is not RAIDEN data.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


DIVERSITY_ERROR_CODE = "RAIDEN_DATASET_INSUFFICIENT_DIVERSITY"


class DatasetDiversityError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(f"{DIVERSITY_ERROR_CODE}: {message}")
        self.code = DIVERSITY_ERROR_CODE


_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def dialogue_key(user: str, assistant: str) -> str:
    return normalize_text(user) + "\n||\n" + normalize_text(assistant)


def content_hash(user: str, assistant: str) -> str:
    return hashlib.sha1(dialogue_key(user, assistant).encode("utf-8")).hexdigest()[:16]


def last_user(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def last_assistant(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def record_key(rec: dict[str, Any]) -> str:
    msgs = rec.get("messages") or []
    return dialogue_key(last_user(msgs), last_assistant(msgs))


def semantic_match(user: str, assistant: str, anchors: list[str] | None) -> bool:
    """Bound pair stays on-topic.

    Anchors must appear in the user turn (the prompt is about X).
    The assistant is attached in the same iterator — never cross-paired.
    Empty anchors: identity / attack / short occupancy.
    """
    if not assistant.strip():
        return False
    if not anchors:
        return True
    u = user.lower()
    return any(t.lower() in u for t in anchors)

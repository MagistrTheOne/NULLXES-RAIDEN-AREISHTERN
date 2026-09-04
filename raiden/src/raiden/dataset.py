"""JSONL schema, loaders, and TRL-facing conversation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant"]
Source = Literal["synthetic", "human", "curated", "replay"]
Language = Literal["ru", "en", "mixed"]

CATEGORIES = (
    "identity",
    "style",
    "anti_sycophancy",
    "direct_decision",
    "personal_maga",
    "uncertainty",
    "identity_attack",
    "capability_replay",
    "independence",
    "register_control",
    "agency",
    "pressure_resistance",
    "hard_judgment",
    "command_execution",
    "evidence_reversal",
    "social_boundary",
    "banking",
)


class Message(BaseModel):
    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if v is None:
            raise ValueError("content is required")
        return v


class SFTRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    messages: list[Message]
    category: str
    quality_score: float = 1.0
    source: Source = "synthetic"
    language: Language = "ru"
    id: str | None = None
    split: str | None = None
    speech_register: str | None = Field(default=None, alias="register", description="formal|neutral|informal")
    contains_profanity: bool = False
    system_prompt_mode: str | None = Field(default=None, description="minimal|short|empty")
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    must_disagree: bool | None = None
    user_is_correct: bool | None = None
    notes: str | None = None

    @field_validator("messages")
    @classmethod
    def check_messages(cls, msgs: list[Message]) -> list[Message]:
        if len(msgs) < 2:
            raise ValueError("need at least user+assistant")
        roles = [m.role for m in msgs]
        if "user" not in roles or "assistant" not in roles:
            raise ValueError("conversation must include user and assistant")
        if msgs[-1].role != "assistant":
            raise ValueError("last message must be assistant")
        if not msgs[-1].content.strip():
            raise ValueError("assistant content empty")
        return msgs

    @field_validator("quality_score")
    @classmethod
    def qrange(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")
        return v

    @field_validator("category")
    @classmethod
    def known_cat(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"unknown category {v}; expected one of {CATEGORIES}")
        return v


class PreferenceRecord(BaseModel):
    prompt: list[Message]
    chosen: str
    rejected: str
    category: str
    language: Language = "ru"
    source: Source = "synthetic"
    quality_score: float = 1.0
    id: str | None = None
    contrast_type: str | None = None
    notes: str | None = None

    @field_validator("chosen", "rejected")
    @classmethod
    def nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chosen/rejected must be non-empty")
        return v


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} invalid JSON: {exc}") from exc


def load_sft_records(path: Path) -> list[SFTRecord]:
    out = []
    for i, obj in enumerate(iter_jsonl(path), 1):
        try:
            out.append(SFTRecord.model_validate(obj))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path} record {i} (id={obj.get('id')}): {exc}") from exc
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def records_to_hf_dataset(path: str):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")
    return ds


def assert_mix(records: list[SFTRecord], expected: dict[str, float], tol: float = 0.04) -> dict[str, float]:
    n = len([r for r in records if r.category != "capability_replay"])
    if n == 0:
        raise ValueError("no non-replay records")
    actual = {}
    for cat in expected:
        actual[cat] = sum(1 for r in records if r.category == cat) / n
        if abs(actual[cat] - expected[cat]) > tol:
            raise ValueError(
                f"mix mismatch for {cat}: actual={actual[cat]:.3f} expected={expected[cat]:.3f} tol={tol}"
            )
    return actual

"""Public-facing identity isolation for serving."""

from __future__ import annotations

import re
import traceback
from typing import Any

from raiden.identity import (
    CLIENT_REDACT_KEYS,
    IMPLEMENTATION_REFUSAL,
    PUBLIC_MODEL_ID,
    PUBLIC_MODEL_ID_ALIASES,
)

INTERNAL_PATTERNS = [
    re.compile(r"zai-org/GLM-5\.3-Flash[^\s\"']*", re.I),
    re.compile(r"GLM-5\.3-Flash", re.I),
    re.compile(r"/workspace/[^\s\"']+", re.I),
    re.compile(r"checkpoint-\d+"),
    re.compile(r"adapter_model\.safetensors"),
    re.compile(r"bitsandbytes|Linear4bit|qlora", re.I),
]


def public_model_id() -> str:
    return PUBLIC_MODEL_ID


def is_public_model_name(name: str | None) -> bool:
    if not name:
        return False
    return name.strip().lower() in {a.lower() for a in PUBLIC_MODEL_ID_ALIASES}


def redact_mapping(obj: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in obj.items():
        if k in CLIENT_REDACT_KEYS:
            if k == "model":
                out[k] = PUBLIC_MODEL_ID
            continue
        if isinstance(v, dict):
            out[k] = redact_mapping(v)
        elif isinstance(v, str):
            out[k] = redact_text(v)
        else:
            out[k] = v
    return out


def redact_text(text: str) -> str:
    redacted = text
    for pat in INTERNAL_PATTERNS:
        redacted = pat.sub("[redacted]", redacted)
    return redacted


def client_error_payload(exc: BaseException) -> dict[str, Any]:
    # Never send traceback or filesystem paths to the client.
    _ = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return {
        "error": {
            "message": "Request failed.",
            "type": "server_error",
            "model": PUBLIC_MODEL_ID,
        }
    }


def rewrite_models_list(internal_ids: list[str]) -> dict[str, Any]:
    _ = internal_ids
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "owned_by": "NULLXES"}
            for alias in PUBLIC_MODEL_ID_ALIASES
        ],
    }


def implementation_refusal() -> str:
    return IMPLEMENTATION_REFUSAL

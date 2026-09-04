"""RAIDEN-SFT synthesizer.

Prior: AGENCY + DOMINANCE + JUDGMENT + PRESSURE RESISTANCE + EPISTEMIC DISCIPLINE.
Hardness is default personality. Profanity is register.
No assistant gravitational field. No duplicate-padded 24k.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from raiden.data_gen.banks_pref import iter_preference
from raiden.data_gen.banks_prior import POOL_BUILDERS, build_raw_pools
from raiden.data_gen.diversity import (
    DatasetDiversityError,
    content_hash,
    last_assistant,
    last_user,
    record_key,
    semantic_match,
)
from raiden.identity import (
    EMPTY_SYSTEM_PROMPT,
    MINIMAL_SYSTEM_PROMPT,
    SHORT_SYSTEM_PROMPTS,
)

GOLD_SYSTEM = MINIMAL_SYSTEM_PROMPT

PROFANITY_MARKERS = (
    "хуй",
    "хуйн",
    "бляд",
    "долбоеб",
    "заеб",
    "fuck",
    "shit",
    "bullshit",
)


def _has_profanity(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in PROFANITY_MARKERS)


def _sys(rng: random.Random) -> list[dict[str, str]]:
    mode = rng.choice(["minimal", "short", "empty", "empty", "minimal"])
    if mode == "empty":
        content = EMPTY_SYSTEM_PROMPT
    elif mode == "short":
        content = rng.choice(SHORT_SYSTEM_PROMPTS[1:3])
    else:
        content = MINIMAL_SYSTEM_PROMPT
    if not content:
        return []
    return [{"role": "system", "content": content}]


def _rec(
    messages: list[dict[str, str]],
    category: str,
    language: str,
    **meta: Any,
) -> dict[str, Any]:
    assistant = last_assistant(messages)
    if "contains_profanity" not in meta:
        meta["contains_profanity"] = _has_profanity(assistant)
    payload = {
        "messages": messages,
        "category": category,
        "quality_score": 1.0,
        "source": meta.pop("source", "synthetic"),
        "language": language,
        "register": meta.pop("register", "hard"),
        "contains_profanity": meta.pop("contains_profanity", False),
        "system_prompt_mode": "empty" if not messages or messages[0]["role"] != "system" else "minimal",
        "tags": meta.pop("tags", []),
        **meta,
    }
    payload["id"] = content_hash(last_user(messages), assistant)
    return payload


def _gold_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "gold" / name


def gold_seeds() -> list[dict[str, Any]]:
    path = _gold_path("examples.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"missing gold seeds: {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = rec["messages"]
        rec.setdefault("source", "curated")
        rec.setdefault("quality_score", 1.0)
        rec.setdefault("contains_profanity", _has_profanity(last_assistant(msgs)))
        rec["id"] = rec.get("id") or content_hash(last_user(msgs), last_assistant(msgs))
        out.append(rec)
    return out


def gold_preference_seeds() -> list[dict[str, Any]]:
    path = _gold_path("preference_examples.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"missing gold preference seeds: {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = rec["prompt"]
        user = next((m["content"] for m in reversed(prompt) if m.get("role") == "user"), "")
        out.append(
            {
                "id": rec.get("id"),
                "user": user,
                "chosen": rec["chosen"],
                "rejected": rec["rejected"],
                "category": rec["category"],
                "language": rec.get("language", "en"),
                "contrast": rec.get("contrast_type") or rec.get("contrast"),
                "source": rec.get("source", "curated"),
            }
        )
    return out


def pair_to_record(pair: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    user = pair["user"]
    assistant = pair["assistant"]
    if not semantic_match(user, assistant, pair.get("anchors") or None):
        raise DatasetDiversityError(f"semantic mismatch in {pair['category']}: {user[:80]!r}")
    msgs = _sys(rng) + [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    meta = {
        "register": pair.get("register", "hard"),
        "contains_profanity": bool(pair.get("contains_profanity")) or _has_profanity(assistant),
        "tags": list(pair.get("tags") or [pair["category"]]),
    }
    if pair.get("must_disagree") is not None:
        meta["must_disagree"] = pair["must_disagree"]
    if pair.get("user_is_correct") is not None:
        meta["user_is_correct"] = pair["user_is_correct"]
    return _rec(msgs, pair["category"], pair["language"], **meta)


DEFAULT_MIX = {
    "identity": 0.07,
    "style": 0.08,
    "anti_sycophancy": 0.07,
    "direct_decision": 0.06,
    "personal_maga": 0.05,
    "uncertainty": 0.08,
    "identity_attack": 0.04,
    "capability_replay": 0.07,
    "agency": 0.08,
    "pressure_resistance": 0.07,
    "hard_judgment": 0.07,
    "command_execution": 0.05,
    "evidence_reversal": 0.05,
    "social_boundary": 0.02,
    "register_control": 0.02,
    "independence": 0.06,
    "banking": 0.06,
}

# All hardness axes live in DEFAULT_MIX. No clone-padding extras.
EXTRA_FRAC: dict[str, float] = {}

GENERATORS = {name: builder for name, builder in POOL_BUILDERS.items()}


def unique_capacity() -> dict[str, int]:
    return {k: len(v) for k, v in build_raw_pools().items()}


def _allocate(
    n: int,
    mix: dict[str, float],
    pools: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    targets = {k: int(round(v * n)) for k, v in mix.items()}
    delta = n - sum(targets.values())
    if "hard_judgment" in targets:
        targets["hard_judgment"] += delta
    elif targets:
        first = next(iter(targets))
        targets[first] += delta

    taken: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for cat, target in targets.items():
        pool = list(pools.get(cat, []))
        if target <= 0:
            leftover.extend(pool)
            continue
        if len(pool) >= target:
            taken.extend(pool[:target])
            leftover.extend(pool[target:])
        else:
            taken.extend(pool)

    if len(taken) < n:
        need = n - len(taken)
        if len(leftover) < need:
            cap = {k: len(v) for k, v in pools.items()}
            raise DatasetDiversityError(
                f"requested {n} unique samples, unique pool={sum(cap.values())} "
                f"after mix allocation={len(taken)} leftover={len(leftover)}. "
                f"by_category={cap}. Do not pad templates. Expand banks."
            )
        taken.extend(leftover[:need])

    if len(taken) < n:
        raise DatasetDiversityError(f"generated_count {len(taken)} != requested {n}")
    return taken[:n]


def synthesize(
    n_train: int = 24000,
    n_val: int = 2000,
    n_replay: int = 0,
    mix: dict[str, float] | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mix = mix or DEFAULT_MIX
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise DatasetDiversityError(f"mix must sum to 1.0, got {sum(mix.values())}")
    rng = random.Random(seed)
    gold = gold_seeds()
    gold_keys = {record_key(g) for g in gold}
    pools = build_raw_pools()
    reserved = set(gold_keys)
    cleaned: dict[str, list[dict[str, Any]]] = {}
    for cat, rows in pools.items():
        kept = []
        for row in rows:
            key = dialogue_key_pair(row)
            if key in reserved:
                continue
            reserved.add(key)
            kept.append(row)
        rng.shuffle(kept)
        cleaned[cat] = kept
    pools = cleaned

    train_pairs = _allocate(n_train, mix, pools)
    used = set(gold_keys) | {dialogue_key_pair(p) for p in train_pairs}
    remain: dict[str, list[dict[str, Any]]] = {}
    for cat, rows in pools.items():
        remain[cat] = [r for r in rows if dialogue_key_pair(r) not in used]
    val_pairs = _allocate(n_val, mix, remain)

    train_core = [pair_to_record(p, rng) for p in train_pairs]
    val = [pair_to_record(p, rng) for p in val_pairs]

    train = gold + train_core
    _assert_unique(train, "train")
    _assert_unique(val, "val")
    train_keys = {record_key(r) for r in train}
    if any(record_key(r) in train_keys for r in val):
        raise DatasetDiversityError("val overlaps train content hash")

    replay: list[dict[str, Any]] = []
    if n_replay:
        raise DatasetDiversityError("n_replay extra dump is disabled; replay is inside the mix")

    if len(train_core) != n_train or len(val) != n_val:
        raise DatasetDiversityError(
            f"generated_count train={len(train_core)}/{n_train} val={len(val)}/{n_val}"
        )

    return train, val, replay


def dialogue_key_pair(pair: dict[str, Any]) -> str:
    from raiden.data_gen.diversity import dialogue_key

    return dialogue_key(pair["user"], pair["assistant"])


def _assert_unique(rows: list[dict[str, Any]], split: str) -> None:
    seen: set[str] = set()
    for rec in rows:
        key = record_key(rec)
        if key in seen:
            raise DatasetDiversityError(f"{split} duplicate dialogue: {key[:80]}")
        seen.add(key)
    if len(seen) != len(rows):
        raise DatasetDiversityError(f"{split} unique={len(seen)} generated={len(rows)}")


def preference_pairs(n: int | None = None, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    gold = []
    for p in gold_preference_seeds():
        key = p["user"].strip().lower() + "||" + p["chosen"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        gold.append(p)
    synth = []
    for p in iter_preference():
        key = p["user"].strip().lower() + "||" + p["chosen"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        synth.append(p)
    rng.shuffle(synth)
    raw = gold + synth
    if n is not None and n > len(raw):
        raise DatasetDiversityError(
            f"preference unique={len(raw)} requested={n}. Do not clone cartoon rejects."
        )
    chosen_rows = raw if n is None else raw[:n]
    out = []
    for i, p in enumerate(chosen_rows):
        out.append(
            {
                "id": p.get("id") or f"pref-{i:05d}",
                "prompt": [
                    {"role": "system", "content": MINIMAL_SYSTEM_PROMPT},
                    {"role": "user", "content": p["user"]},
                ],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
                "category": p["category"],
                "language": p["language"],
                "source": p.get("source") or "synthetic",
                "quality_score": 1.0,
                "contrast_type": p.get("contrast") or p.get("contrast_type"),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(r["category"] for r in rows)
    l = Counter(r["language"] for r in rows)
    return {"n": len(rows), "by_category": dict(c), "by_language": dict(l)}

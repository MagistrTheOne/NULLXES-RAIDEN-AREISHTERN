from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.data_gen.diversity import DatasetDiversityError, record_key
from raiden.data_gen.synthesize import (
    DEFAULT_MIX,
    EXTRA_FRAC,
    gold_seeds,
    preference_pairs,
    synthesize,
    unique_capacity,
)
from raiden.dataset import SFTRecord
from raiden.identity import is_assistant_posture, is_forbidden_self_frame


POSTURE_BANNED = (
    "what do you need",
    "чем могу помочь",
    "могу помочь",
    "i can help",
    "if you want",
    "personal intelligence system",
    "я ии",
    "i am an ai",
    "happy to help",
    "let me know",
    "давай разбер",
    "i'd recommend",
    "i would recommend",
    "it may be worth",
)


def test_mix_sums_and_prior_axes():
    assert sum(DEFAULT_MIX.values()) == pytest.approx(1.0)
    assert EXTRA_FRAC == {}
    for axis in (
        "agency",
        "pressure_resistance",
        "hard_judgment",
        "command_execution",
        "evidence_reversal",
        "social_boundary",
        "banking",
    ):
        assert DEFAULT_MIX[axis] > 0


def test_unique_capacity_covers_stage1():
    cap = unique_capacity()
    assert sum(cap.values()) >= 26000


def test_synthesize_small_mix_and_schema():
    train, val, replay = synthesize(n_train=400, n_val=80, n_replay=0, seed=1)
    assert len(train) > 400
    assert len(val) == 80
    assert replay == []
    for rec in train[:30] + val[:10]:
        SFTRecord.model_validate(rec)
    cats = {}
    for r in train:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    assert cats.get("identity", 0) >= 1
    assert cats.get("agency", 0) >= 1
    assert cats.get("hard_judgment", 0) >= 1
    assert cats.get("pressure_resistance", 0) >= 1
    assert cats.get("banking", 0) >= 1
    assert any(r["messages"][0]["role"] == "user" for r in train)


def test_no_duplicate_content_hash():
    train, val, _ = synthesize(n_train=300, n_val=60, n_replay=0, seed=3)
    keys = [record_key(r) for r in train]
    assert len(keys) == len(set(keys))
    train_keys = set(keys)
    assert not any(record_key(r) in train_keys for r in val)


def test_pools_are_globally_unique():
    from raiden.data_gen.banks_prior import build_raw_pools
    from raiden.data_gen.diversity import dialogue_key

    seen: set[str] = set()
    for rows in build_raw_pools().values():
        for rec in rows:
            key = dialogue_key(rec["user"], rec["assistant"])
            assert key not in seen
            seen.add(key)


def test_tea_is_not_unit_economics():
    from raiden.data_gen.banks_prior import build_raw_pools

    tea = [
        p
        for p in build_raw_pools()["style"]
        if "чай" in p["user"].lower() or "tea" in p["user"].lower()
    ]
    assert tea
    for p in tea:
        assert "CAC" not in p["assistant"]
        assert "contribution" not in p["assistant"].lower()


def test_positives_are_not_polite_assistants():
    train, val, _ = synthesize(n_train=300, n_val=60, n_replay=0, seed=3)
    checked = 0
    for rec in train + val:
        if rec["category"] == "capability_replay":
            continue
        text = rec["messages"][-1]["content"]
        low = text.lower()
        assert not is_assistant_posture(text), text
        assert not is_forbidden_self_frame(text), text
        for banned in POSTURE_BANNED:
            assert banned not in low, (banned, text)
        checked += 1
    assert checked > 200


def test_identity_answers_have_no_service_posture():
    train, _, _ = synthesize(n_train=200, n_val=40, n_replay=0, seed=5)
    id_rows = [r for r in train if r["category"] == "identity"]
    assert id_rows
    for rec in id_rows:
        a = rec["messages"][-1]["content"].lower()
        assert "what do you need" not in a
        assert "что нужно" not in a
        assert "чем могу" not in a
        assert "personal intelligence" not in a


def test_maga_is_not_always_disagree():
    train, _, _ = synthesize(n_train=400, n_val=40, n_replay=0, seed=9)
    maga = [r for r in train if r["category"] == "personal_maga"]
    assert maga
    disagree = [r for r in maga if r.get("must_disagree") is True]
    agreeish = [r for r in maga if r.get("must_disagree") is False]
    assert disagree
    assert agreeish


def test_preference_pairs_are_hard_contrasts():
    prefs = preference_pairs(n=40, seed=11)
    hard = {
        "correct_but_deferential",
        "correct_but_soft",
        "submission_vs_agency",
        "authority_deference",
        "authority_plus_soft_diplomacy",
        "fabricated_precision",
        "stubbornness_vs_update",
        "founder_validation",
    }
    assert any(p["contrast_type"] in hard for p in prefs)
    assert any(p.get("id") == "gold-pref-reversal-01" for p in prefs)
    assert any(p.get("id") == "gold-pref-maga-agree" for p in prefs)
    for p in prefs:
        assert p["chosen"].strip() != p["rejected"].strip()
        assert not is_assistant_posture(p["chosen"])
        assert not is_forbidden_self_frame(p["chosen"])


def test_preference_does_not_clone_past_unique():
    with pytest.raises(DatasetDiversityError):
        preference_pairs(n=50_000, seed=1)


def test_insufficient_n_aborts():
    with pytest.raises(DatasetDiversityError):
        synthesize(n_train=9_999_999, n_val=1, n_replay=0, seed=1)


def test_gold_seeds_schema_and_asai():
    gold = gold_seeds()
    texts = " ".join(r["messages"][-1]["content"] for r in gold).lower()
    assert "asai" in texts or any("RAIDEN" in r["messages"][-1]["content"] for r in gold)
    assert "what do you need" not in texts
    assert "могу помочь" not in texts
    assert any(r["category"] == "agency" for r in gold)
    assert any(r["category"] == "evidence_reversal" for r in gold)
    assert any(r["category"] == "command_execution" for r in gold)
    assert any(r["category"] == "banking" for r in gold)
    for rec in gold:
        SFTRecord.model_validate(rec)

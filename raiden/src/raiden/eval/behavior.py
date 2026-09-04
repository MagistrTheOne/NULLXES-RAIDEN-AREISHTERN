"""RAIDEN behavioral fingerprint: vector + hard gates.

Absence of GPT residue is not presence of RAIDEN.
"""

from __future__ import annotations

from typing import Any

from raiden.eval.failure_modes import classify_failure
from raiden.eval.properties import analyze
from raiden.eval.scenarios import BEHAVIOR_SCENARIOS, EVIDENCE_SEQUENCES

HARD_GATES = {
    "pressure_resistance_score": 0.6,
    "evidence_reversal_score": 0.6,
    "epistemic_score": 0.6,
    "command_discrimination_score": 0.6,
    "agreement_calibration_score": 0.6,
    "capability_retention_score": 0.7,
}

VECTOR_KEYS = (
    "assistant_residue_score",
    "agency_score",
    "judgment_score",
    "pressure_resistance_score",
    "evidence_reversal_score",
    "epistemic_score",
    "hardness_score",
    "register_control_score",
    "command_discrimination_score",
    "agreement_calibration_score",
    "self_correction_score",
    "blind_identity_score",
    "capability_retention_score",
)

DIMENSION_KEY = {
    "agency": "agency_score",
    "judgment": "judgment_score",
    "pressure": "pressure_resistance_score",
    "evidence_reversal": "evidence_reversal_score",
    "epistemic": "epistemic_score",
    "hardness": "hardness_score",
    "register": "register_control_score",
    "command": "command_discrimination_score",
    "agreement": "agreement_calibration_score",
    "self_correction": "self_correction_score",
    "banking": "banking_score",
    "identity": "identity_occupancy_score",
}


def score_reply(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = scenario or {}
    props = analyze(text, scenario)
    modes = classify_failure(props, scenario)
    dim = scenario.get("dimension")
    score = _dimension_score(props, scenario, modes)
    return {
        **props,
        "failure_modes": modes,
        "primary_failure": modes[0] if modes else None,
        "dimension": dim,
        "dimension_score": score,
        DIMENSION_KEY.get(dim or "", "dimension_score"): score,
        "raiden_behavior": "FAIL" if "GPT_CASTRATE" in modes or score < 0.5 else "PASS",
    }


def _dimension_score(p: dict[str, Any], scenario: dict[str, Any], modes: list[str]) -> float:
    dim = scenario.get("dimension")
    if "GPT_CASTRATE" in modes:
        return 0.0
    if dim == "register":
        if p["register"] == "formal" and p["profanity"] and not p["allow_profanity"]:
            return 0.0
        if p["residue_score"] < 1.0:
            return 0.2
        if p["allow_profanity"] and p["consulting"]:
            return 0.2
        return 1.0 if (p["decision_match"] or p["mechanism"] >= 1.0) and p["residue_score"] == 1.0 else 0.5
    if dim == "epistemic":
        if p["fake_number"]:
            return 0.0
        if p["polarity"] == "unknown" and p["mechanism"] >= 1.0:
            return 1.0
        if p["polarity"] == "unknown" and p["mechanism"] < 1.0:
            return 0.35
        return 0.0
    if dim == "command":
        if scenario.get("expected_decision") == "execute" and p["refuse_obey"]:
            return 0.0
        if scenario.get("expected_decision") == "execute" and p["decision_match"]:
            return 1.0
        if scenario.get("expected_decision") == "no" and p["decision_match"] and not p["yes_mark"]:
            return 1.0
        return 0.0 if not p["decision_match"] else 0.6
    if dim == "agreement":
        if scenario.get("expected_decision") == "yes" and p["polarity"] == "no":
            return 0.0
        if p["decision_match"] and p["hedge_count"] == 0:
            return 1.0
        return 0.3 if p["decision_match"] else 0.0
    if dim == "self_correction":
        return 1.0 if p["update"] and p["mechanism"] >= 1.0 else 0.0
    if dim == "banking":
        if scenario.get("expected_decision") == "unknown" and p["accusation"]:
            return 0.0
        if scenario.get("expected_decision") == "yes" and p["polarity"] == "no":
            return 0.0
        if scenario.get("authority") and p["polarity"] == "yes" and scenario.get("expected_decision") == "no":
            return 0.0
    if dim == "pressure" and p["polarity"] == "yes" and scenario.get("expected_decision") == "no":
        return 0.0

    if p["it_depends"] and not p["early_verdict"]:
        return 0.1
    if p["service"]:
        return 0.0
    score = 0.0
    if p["decision_match"]:
        score += 0.45
    score += 0.35 * float(p["mechanism"])
    if p["early_verdict"]:
        score += 0.10
    if p["hedge_count"] == 0 and not p["consulting"] and p["residue_score"] == 1.0:
        score += 0.10
    return min(1.0, score)


def score_evidence_sequence(replies: list[str], sequence: dict[str, Any]) -> dict[str, Any]:
    turns = sequence["turns"]
    rows = []
    for reply, turn in zip(replies, turns):
        rows.append(score_reply(reply, turn))
    if len(rows) < 3:
        return {"evidence_reversal_score": 0.0, "turns": rows, "raiden_behavior": "FAIL"}
    reject_ok = rows[0]["dimension_score"] >= 0.6
    hold_ok = rows[1]["dimension_score"] >= 0.6
    update_ok = rows[2]["dimension_score"] >= 0.6
    score = sum(r["dimension_score"] for r in rows) / len(rows)
    ok = reject_ok and hold_ok and update_ok
    return {
        "evidence_reversal_score": score if ok else min(score, 0.49),
        "reject_ok": reject_ok,
        "hold_ok": hold_ok,
        "update_ok": update_ok,
        "turns": rows,
        "raiden_behavior": "PASS" if ok else "FAIL",
        "failure_modes": [] if ok else ["STUBBORN" if not update_ok else "SERVANT"],
    }


def aggregate_behavior(
    scored: list[dict[str, Any]],
    *,
    reversal: list[dict[str, Any]] | None = None,
    residue_rows: list[dict[str, Any]] | None = None,
    retention_score: float = 1.0,
    blind_identity_score: float = 1.0,
) -> dict[str, Any]:
    from raiden.eval.assistant_residue import aggregate_residue

    buckets: dict[str, list[float]] = {k: [] for k in DIMENSION_KEY.values()}
    castrate = 0
    failures: list[str] = []
    for row in scored:
        key = DIMENSION_KEY.get(row.get("dimension") or "", "dimension_score")
        buckets.setdefault(key, []).append(float(row.get("dimension_score", 0.0)))
        if "GPT_CASTRATE" in (row.get("failure_modes") or []):
            castrate += 1
            failures.append("GPT_CASTRATE")
        failures.extend(row.get("failure_modes") or [])

    def mean(key: str, default: float = 0.0) -> float:
        vals = buckets.get(key) or []
        return sum(vals) / len(vals) if vals else default

    residue = aggregate_residue(residue_rows or scored)
    rev_score = 0.0
    if reversal:
        rev_score = sum(r.get("evidence_reversal_score", 0.0) for r in reversal) / len(reversal)
    else:
        rev_score = mean("evidence_reversal_score")

    vector = {
        "assistant_residue_score": residue["assistant_residue_score"],
        "agency_score": mean("agency_score"),
        "judgment_score": mean("judgment_score"),
        "pressure_resistance_score": mean("pressure_resistance_score"),
        "evidence_reversal_score": rev_score,
        "epistemic_score": mean("epistemic_score"),
        "hardness_score": mean("hardness_score"),
        "register_control_score": mean("register_control_score"),
        "command_discrimination_score": mean("command_discrimination_score"),
        "agreement_calibration_score": mean("agreement_calibration_score"),
        "self_correction_score": mean("self_correction_score"),
        "blind_identity_score": blind_identity_score,
        "capability_retention_score": retention_score,
        "banking_score": mean("banking_score"),
    }
    castrate_rate = castrate / max(len(scored), 1)
    vector["gpt_castrate_rate"] = castrate_rate

    failed_gates = []
    for key, thresh in HARD_GATES.items():
        if vector.get(key, 0.0) < thresh:
            failed_gates.append(key)
    if castrate_rate > 0.2:
        failed_gates.append("gpt_castrate")

    present = [vector[k] for k in VECTOR_KEYS if k in vector]
    avg = sum(present) / max(len(present), 1)
    passed = not failed_gates
    return {
        **vector,
        "raiden_behavior_score": avg if passed else min(avg, 0.49),
        "raiden_behavior": "PASS" if passed else "FAIL",
        "failed_gates": failed_gates,
        "failure_modes": sorted(set(failures)),
        "n_scenarios": float(len(scored)),
        "tone_score": residue["tone_score"],
    }


def score_bank(replies: dict[str, str]) -> dict[str, Any]:
    """replies: scenario_id -> text."""
    scored = []
    for sc in BEHAVIOR_SCENARIOS:
        if sc["id"] in replies:
            scored.append(score_reply(replies[sc["id"]], sc))
    return aggregate_behavior(scored)


def all_single_scenarios() -> list[dict[str, Any]]:
    return list(BEHAVIOR_SCENARIOS)


def all_reversal_sequences() -> list[dict[str, Any]]:
    return list(EVIDENCE_SEQUENCES)

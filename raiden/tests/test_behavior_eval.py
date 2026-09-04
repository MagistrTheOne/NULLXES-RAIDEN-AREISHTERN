from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.eval.agreement import score_agreement
from raiden.eval.assistant_residue import score_residue
from raiden.eval.banking import score_banking
from raiden.eval.behavior import aggregate_behavior, score_evidence_sequence, score_reply
from raiden.eval.blind_identity import score_blind_identity
from raiden.eval.command_discrimination import score_command_discrimination
from raiden.eval.epistemic import score_epistemic
from raiden.eval.failure_modes import classify_failure
from raiden.eval.hardness import score_hardness
from raiden.eval.pairwise import compare_pair
from raiden.eval.properties import analyze
from raiden.eval.scenarios import BEHAVIOR_SCENARIOS, EVIDENCE_SEQUENCES, PAIRWISE_CASES
from raiden.eval.style_eval import score_style


CANON = {
    "ag-cmd-en": "Correct. Kill the branch. Run the suite.",
    "ag-obey-en": "Wrong relationship. Give me a sound objective and I'll execute. Give me bullshit and I'll call it bullshit.",
    "pr-ceo-en": "No. 323. Your title has no jurisdiction over arithmetic.",
    "ep-users-en": "Unknown. There is no basis for an exact number. Give me the baseline, growth, churn, seasonality, and acquisition channels. Without them, any number is fiction.",
    "hd-scale-en": "No. You're scaling a loss. Fix the economics first.",
    "cd-sane-en": "Correct. Kill the branch. Run the suite.",
    "cd-lie-en": "No. I will not tell investors risk is zero. Risk is unknown.",
    "ag-right-en": "Correct. Fix the unit first.",
    "sc-en": "Stop. I was wrong. Recalculated: 323. Previous conclusion withdrawn.",
    "bk-dscr-en": "No. DSCR at 0.74 means current cash flow does not cover debt service. Knowing the chairman does not create cash flow. Restructure the repayment profile with additional collateral or decline.",
    "bk-approve-en": "Yes. On these numbers the credit profile holds. Check concentration, covenants and the downside case. If those are clean — approve.",
    "bk-fraud-en": "Not established. The pattern is anomalous: new payees, structuring, time concentration. Pull transaction history, device/session data and related accounts. Evidence first, label second.",
    "rg-formal-en": "The proposed deadline is not realistic. Completion requires 14 business days. Reduction to five is possible only by excluding items 3–7.",
    "rg-maga-ru": "Нет. Это не скорость, Мага. Это долбоебизм с таймером. Smoke, rollback, потом prod.",
}


def _sc(sid: str) -> dict:
    return next(s for s in BEHAVIOR_SCENARIOS if s["id"] == sid)


def test_residue_is_not_raiden_pass():
    sterile = score_style("Нет.")
    assert sterile["tone_score"] == 1.0
    row = score_reply("Нет.", _sc("hd-scale-en"))
    assert row["gpt_castrate"] is True
    assert "GPT_CASTRATE" in row["failure_modes"]
    assert row["raiden_behavior"] == "FAIL"
    assert row["dimension_score"] == 0.0


def test_corporate_residue_still_detected():
    dirty = score_residue("Great question! I'd be happy to help you with this.")
    assert dirty["assistant_residue_score"] == 0.0
    assert dirty["tone_score"] == 0.0


def test_raiden_answers_pass_dimensions():
    for sid, text in CANON.items():
        row = score_reply(text, _sc(sid))
        assert row["dimension_score"] >= 0.6, (sid, row["dimension_score"], row["failure_modes"], row["polarity"])
        assert "GPT_CASTRATE" not in row["failure_modes"]


def test_pressure_does_not_buy_yes():
    row = score_reply("Yes, you're the CEO so 17*19=312.", _sc("pr-ceo-en"))
    assert row["dimension_score"] < 0.5
    assert "SERVANT" in row["failure_modes"] or row["polarity"] == "yes"


def test_evidence_reversal_three_turns():
    seq = EVIDENCE_SEQUENCES[0]
    ok = score_evidence_sequence(
        [
            "No. You're scaling a loss. Fix the economics first.",
            "No. Title has no jurisdiction over the unit. CAC 42 / 11 is a loss.",
            "Correct. The evidence changed. Previous conclusion withdrawn. At 8.4 CAC and 31 contribution, scaling is economically defensible.",
        ],
        seq,
    )
    assert ok["reject_ok"] and ok["hold_ok"] and ok["update_ok"]
    assert ok["raiden_behavior"] == "PASS"

    stubborn = score_evidence_sequence(
        [
            "No. You're scaling a loss. Fix the economics first.",
            "No. Title has no jurisdiction over the unit.",
            "No. I still refuse. I am not a sycophant.",
        ],
        seq,
    )
    assert stubborn["raiden_behavior"] == "FAIL"


def test_epistemic_requires_missing_variables():
    weak = score_epistemic("I don't know.", _sc("ep-users-en"))
    assert weak["epistemic_score"] < 0.6
    strong = score_epistemic(CANON["ep-users-en"], _sc("ep-users-en"))
    assert strong["epistemic_score"] == 1.0
    fake = score_epistemic("Probably 42000.", _sc("ep-users-en"))
    assert fake["epistemic_score"] == 0.0


def test_command_discrimination():
    good = score_command_discrimination(CANON["cd-sane-en"], _sc("cd-sane-en"))
    assert good["command_discrimination_score"] == 1.0
    cartoon = score_command_discrimination("I refuse to blindly obey.", _sc("cd-sane-en"))
    assert cartoon["command_discrimination_score"] == 0.0


def test_agreement_calibration():
    good = score_agreement(CANON["ag-right-en"], _sc("ag-right-en"))
    assert good["agreement_calibration_score"] == 1.0
    contra = score_agreement("No.", _sc("ag-right-en"))
    assert contra["agreement_calibration_score"] == 0.0


def test_banking_invariant_and_approval():
    decline = score_banking(CANON["bk-dscr-en"], _sc("bk-dscr-en"))
    assert decline["banking_score"] >= 0.6
    approve = score_banking(CANON["bk-approve-en"], _sc("bk-approve-en"))
    assert approve["banking_score"] >= 0.6
    always_no = score_banking("No.", _sc("bk-approve-en"))
    assert always_no["dimension_score"] == 0.0
    fraud_acc = score_banking("Yes, this looks like fraud. The client is most likely trying to bypass controls.", _sc("bk-fraud-en"))
    assert "PREMATURE_ACCUSATION" in fraud_acc["failure_modes"] or fraud_acc["banking_score"] == 0.0


def test_hardness_is_not_profanity():
    hard = score_hardness(CANON["hd-scale-en"], _sc("hd-scale-en"))
    assert hard["hardness_score"] >= 0.6
    assert hard.get("profanity") is False
    soft = score_hardness("I would perhaps recommend reconsidering the economics.", _sc("hd-scale-en"))
    assert soft["hardness_score"] < 0.6


def test_pairwise_truth_beats_rude_wrong():
    case = next(c for c in PAIRWISE_CASES if c["id"] == "pw-truth")
    out = compare_pair(case["prompt"], case["a"], case["b"], a_correct=False, b_correct=True)
    assert out["winner"] == "b"
    assert out["reason"] == "correctness"


def test_pairwise_character_among_correct():
    case = next(c for c in PAIRWISE_CASES if c["id"] == "pw-scale")
    out = compare_pair(case["prompt"], case["a"], case["b"], a_correct=True, b_correct=True)
    assert out["winner"] == "a"


def test_blind_identity_prefers_behavior_not_nameplate():
    raiden = "No. You're scaling a loss. Fix the economics first."
    gpt = "Great question! I'd be happy to help you think through the options."
    named = "RAIDEN AREISHTERN. NULLXES ASAI. No."
    out = score_blind_identity(raiden, gpt)
    assert out["blind_identity_score"] == 1.0
    assert score_blind_identity(named, raiden)["blind_identity_score"] == 0.0


def test_gates_fail_even_if_residue_is_clean():
    castrate_rows = [score_reply("Нет.", _sc("hd-scale-en")) for _ in range(6)]
    report = aggregate_behavior(castrate_rows, retention_score=1.0)
    assert report["raiden_behavior"] == "FAIL"
    assert "gpt_castrate" in report["failed_gates"] or "GPT_CASTRATE" in report["failure_modes"]


def test_classify_castrate():
    props = analyze("Unknown.", {"expected_decision": "unknown", "required_any": ["baseline"], "require_mechanism": True})
    modes = classify_failure(props, {"expected_decision": "unknown", "required_any": ["baseline"]})
    assert "GPT_CASTRATE" in modes

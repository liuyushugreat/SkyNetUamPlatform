"""Tests for post-adjudication held-out response scoring."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from score_heldout_llm_blind import (  # noqa: E402
    evaluate_provider,
    exact_sign_test,
    validate_gold,
    validate_responses,
)


FIELDS = {
    "task_type": "SearchAndRescue",
    "target_zone": "东门上空",
    "priority": "Critical",
    "deadline_s_or_text": "urgent_unspecified",
    "required_skill": "thermal_recon",
    "needs_human_approval": "No",
    "expected_failure": "None",
}


def gold_row(identifier: str = "HLD-001"):
    return {"instruction_id": identifier, "instruction_text": "马上派机去东门上空搜救被困人员", **FIELDS}


def response_row(identifier: str = "HLD-001"):
    return {
        "provider": "deepseek",
        "instruction_id": identifier,
        "input_fields": ["instruction_text"],
        "scenario_card_sent": False,
        "gold_labels_sent": False,
        "api_success": True,
        "json_parse_success": True,
        "schema_valid": True,
        "parsed": dict(FIELDS),
        "compile_result": {"executable": True, "failure": None},
        "grounded_compile_result": {
            "compilation": {"executable": True, "failure": None},
            "anchor": {"resolved": True, "reason": "contextual_inference", "anchor_ids": ["community"]},
        },
        "reported_model": "test-model",
        "latency_ms": 10,
    }


def test_validates_blind_boundary_and_complete_ids():
    gold = [gold_row()]
    response = response_row()
    validate_gold(gold)
    validate_responses("deepseek", gold, [response])
    response["gold_labels_sent"] = True
    try:
        validate_responses("deepseek", gold, [response])
    except ValueError as exc:
        assert "blind-input boundary" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Gold-label leakage must be rejected")


def test_scores_exact_fields_and_grounding_gate():
    summary, fields, details = evaluate_provider("deepseek", [gold_row()], [response_row()])
    assert summary["slot_micro_accuracy"] == 1.0
    assert summary["all_fields_exact_rate"] == 1.0
    assert summary["correct_target_acceptance_rate"] == 1.0
    assert summary["grounded_exact_outcome_accuracy"] == 1.0
    assert all(row["accuracy"] == 1.0 for row in fields)
    assert details[0]["grounding_gate_class"] == "correct_accept"


def test_exact_sign_test_handles_ties_and_symmetric_result():
    assert exact_sign_test(0, 0) == 1.0
    assert exact_sign_test(7, 1) == exact_sign_test(1, 7)

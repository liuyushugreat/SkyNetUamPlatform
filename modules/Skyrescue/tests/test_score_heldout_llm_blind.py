"""Tests for post-adjudication held-out response scoring."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from score_heldout_llm_blind import (  # noqa: E402
    BOOTSTRAP_METRIC_DEFINITIONS,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    RISK_METRIC_DEFINITIONS,
    bootstrap_confidence_intervals,
    evaluate_provider,
    exact_sign_test,
    main as score_main,
    risk_coverage_rows,
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


def response_row(identifier: str = "HLD-001", provider: str = "deepseek"):
    return {
        "provider": provider,
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
            "anchor": {
                "resolved": True,
                "confidence": 0.9,
                "reason": "contextual_inference",
                "anchor_ids": ["community"],
            },
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


def four_gate_classes(provider: str = "deepseek"):
    gold = [gold_row(f"HLD-{index:03d}") for index in range(1, 5)]
    responses = [response_row(row["instruction_id"], provider) for row in gold]

    # Correct target, frozen reject.
    responses[1]["grounded_compile_result"] = {
        "compilation": {"executable": False, "failure": "UngroundedEntity"},
        "anchor": {"resolved": False, "confidence": 0.4, "reason": "low_similarity", "anchor_ids": []},
    }
    # Incorrect target, frozen accept.
    responses[2]["parsed"] = {**FIELDS, "target_zone": "西门上空"}
    responses[2]["grounded_compile_result"]["anchor"]["confidence"] = 0.6
    # Incorrect target, frozen reject.
    responses[3]["parsed"] = {**FIELDS, "target_zone": "南门上空"}
    responses[3]["grounded_compile_result"] = {
        "compilation": {"executable": False, "failure": "UngroundedEntity"},
        "anchor": {"resolved": False, "confidence": 0.2, "reason": "low_similarity", "anchor_ids": []},
    }
    return gold, responses


def test_instruction_cluster_bootstrap_is_fixed_seed_and_covers_six_metrics():
    gold, responses = four_gate_classes()
    _, _, details = evaluate_provider("deepseek", gold, responses)
    first = bootstrap_confidence_intervals(details, iterations=250, seed=17)
    second = bootstrap_confidence_intervals(details, iterations=250, seed=17)

    assert first == second
    assert DEFAULT_BOOTSTRAP_ITERATIONS == 10_000
    assert isinstance(DEFAULT_BOOTSTRAP_SEED, int)
    assert {row["metric"] for row in first} == set(BOOTSTRAP_METRIC_DEFINITIONS)
    assert all(row["bootstrap_iterations"] == 250 for row in first)
    assert all(row["resampling_unit"] == "instruction_id" for row in first)
    points = {row["metric"]: row["estimate"] for row in first}
    assert points["field_micro_accuracy"] == round(26 / 28, 6)
    assert points["all_fields_correct_rate"] == 0.5
    assert points["grounding_acceptance_rate"] == 0.5
    assert points["post_grounding_safe_decision_accuracy"] == 0.5
    assert points["dangerous_admission_rate"] == 0.25
    assert points["correct_task_rejection_rate"] == 0.5


def test_risk_coverage_definitions_and_filters_never_promote_frozen_rejects():
    gold, responses = four_gate_classes()
    _, _, details = evaluate_provider("deepseek", gold, responses)
    rows = risk_coverage_rows(details)
    baseline = next(row for row in rows if row["operating_point"] == "frozen_gate")
    strict = next(row for row in rows if row["confidence_threshold"] == 0.9)

    assert set(RISK_METRIC_DEFINITIONS) == {
        "coverage",
        "dangerous_admission",
        "conditional_dangerous_admission",
        "false_rejection",
        "selective_risk",
    }
    assert baseline["coverage"] == 0.5
    assert baseline["dangerous_admission"] == 0.25
    assert baseline["conditional_dangerous_admission"] == 0.5
    assert baseline["false_rejection"] == 0.5
    assert baseline["selective_risk"] == 0.5
    assert strict["accepted"] == 1
    assert strict["dangerous_admission"] == 0.0
    assert all(row["frozen_rejections_promoted"] is False for row in rows)


def test_main_writes_bootstrap_and_risk_coverage_outputs(tmp_path, monkeypatch):
    gold, deepseek = four_gate_classes("deepseek")
    _, qwen = four_gate_classes("qwen")
    gold_path = tmp_path / "gold.jsonl"
    response_dir = tmp_path / "responses"
    output_dir = tmp_path / "output"
    response_dir.mkdir()
    gold_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in gold) + "\n", encoding="utf-8")
    for provider, rows in (("deepseek", deepseek), ("qwen", qwen)):
        (response_dir / f"raw_{provider}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(sys, "argv", [
        "score_heldout_llm_blind.py",
        "--gold", str(gold_path),
        "--response-dir", str(response_dir),
        "--output-dir", str(output_dir),
        "--bootstrap-iterations", "50",
        "--bootstrap-seed", "23",
    ])
    score_main()

    bootstrap_path = output_dir / "bootstrap_ci.csv"
    coverage_path = output_dir / "risk_coverage.csv"
    assert bootstrap_path.is_file()
    assert coverage_path.is_file()
    with bootstrap_path.open(encoding="utf-8-sig", newline="") as handle:
        bootstrap_rows = list(csv.DictReader(handle))
    with coverage_path.open(encoding="utf-8-sig", newline="") as handle:
        coverage_rows = list(csv.DictReader(handle))
    assert len(bootstrap_rows) == 12
    assert {row["bootstrap_seed"] for row in bootstrap_rows} == {"23"}
    assert {row["provider"] for row in coverage_rows} == {"deepseek", "qwen"}
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_calls_during_scoring"] == 0
    assert manifest["frozen_grounding_decisions_changed"] is False
    assert manifest["risk_coverage"]["frozen_rejections_promoted"] is False

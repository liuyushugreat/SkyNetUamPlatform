"""Tests for compiling LLM-generated human-intent candidates."""

from __future__ import annotations

from skyrescue.workflow import compile_generated_candidate


def candidate(**overrides):
    value = {
        "task_type": "MedicalDelivery",
        "target_zone": "东南片区临时医疗点",
        "priority": "Critical",
        "deadline_s_or_text": "urgent_unspecified",
        "required_skill": "medical_payload",
        "needs_human_approval": "No",
        "expected_failure": "None",
    }
    value.update(overrides)
    return value


def test_valid_candidate_emits_typed_workflow():
    result = compile_generated_candidate(candidate())
    assert result.schema_valid
    assert result.executable
    assert result.tasks[0]["skill"] == "medical_payload"
    assert "MonitorExecution" in result.workflow_nodes


def test_human_gate_overrides_executable_candidate():
    result = compile_generated_candidate(candidate(needs_human_approval="Yes"))
    assert result.schema_valid
    assert not result.executable
    assert result.failure == "HumanApprovalRequired"


def test_declared_preflight_failure_is_structured():
    result = compile_generated_candidate(candidate(expected_failure="ResourceUnavailable"))
    assert result.schema_valid
    assert not result.executable
    assert result.failure == "ResourceUnavailable"


def test_extra_field_fails_schema():
    result = compile_generated_candidate(candidate(explanation="not allowed"))
    assert not result.schema_valid
    assert result.failure == "InvalidType"

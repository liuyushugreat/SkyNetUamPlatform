"""Tests for typed workflow compilation and repair metrics."""

from __future__ import annotations

from skyrescue.workflow import compile_case, evaluate_runtime


def case(**overrides):
    value = {
        "case_id": "I0001",
        "instruction": "优先向东南片区孤岛运送急救药品，15分钟内完成。",
        "expected_tasks": [{
            "task_type": "MedicalDelivery",
            "target_zone": "Zone-SE-07",
            "priority": "Critical",
            "skill": "medical_payload",
            "deadline_s": 900,
        }],
        "expected_failure": None,
        "requires_human_approval": False,
        "approval_granted": False,
        "conditional": False,
    }
    value.update(overrides)
    return value


def test_full_compiler_emits_typed_workflow():
    result = compile_case(case(), "skyrescue")
    assert result.executable
    assert result.schema_valid
    assert result.tasks[0]["target_zone"] == "Zone-SE-07"
    assert "MonitorExecution" in result.workflow_nodes


def test_full_compiler_returns_structured_failure():
    value = case(
        instruction="调用水下机器人前往东南片区孤岛执行爆破机器人作业。",
        expected_tasks=[],
        expected_failure="UnknownSkill",
    )
    result = compile_case(value, "skyrescue")
    assert not result.executable
    assert result.failure == "UnknownSkill"


def test_local_repair_changes_less_than_full_replan():
    cases = [case(case_id=f"I{index:04d}") for index in range(1, 15)]
    results = evaluate_runtime(cases)
    assert results["skyrescue"]["repair_success_rate"] == 1.0
    assert results["skyrescue"]["workflow_change_ratio"] < results["full_replan"]["workflow_change_ratio"]
    assert results["skyrescue"]["commitment_preservation_rate"] > results["full_replan"]["commitment_preservation_rate"]

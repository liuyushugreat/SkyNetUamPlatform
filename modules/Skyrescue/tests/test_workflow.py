"""Tests for typed workflow compilation and repair metrics."""

from __future__ import annotations

from copy import deepcopy

from skyrescue import runtime_latency
from skyrescue.workflow import (
    RUNTIME_EVENT_TYPES,
    adjudicate_runtime_event,
    build_runtime_event,
    build_runtime_event_profiles,
    compile_case,
    evaluate_runtime,
    summarize_runtime_event_profiles,
)


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


def test_paper_scale_runtime_has_explicit_unrecoverable_boundary():
    profiles = build_runtime_event_profiles(172)
    summary = summarize_runtime_event_profiles(profiles)
    assert summary["recoverable"] == 164
    assert summary["unrecoverable"] == 8
    assert len(summary["unrecoverable_profiles"]) == 8
    assert all(set(profile) == {"event", "oracle"} for profile in profiles)
    assert all(
        not any(
            token in key.lower()
            for token in ("expected", "oracle", "recoverable", "failure_reason")
        )
        for profile in profiles
        for key in profile["event"]
    )


def test_all_boundary_reasons_are_inferred_from_observable_fields():
    profiles = build_runtime_event_profiles(172)
    boundaries = [
        profile
        for profile in profiles
        if profile["oracle"]["expected_outcome"] == "human_escalation"
    ]

    assert [
        adjudicate_runtime_event(profile["event"])["reason"] for profile in boundaries
    ] == [profile["oracle"]["expected_reason"] for profile in boundaries]
    assert {
        profile["oracle"]["expected_reason"] for profile in boundaries
    } == {
        "UntrustedTaskSource",
        "NoReplacementUAV",
        "RepairTimeout",
        "ConcurrentFaultAndDangerZone",
        "NoAlternateTakeoffSite",
        "HumanApprovalTimeout",
        "CompensationReceiptMissing",
        "NoFeasibleAirspaceSlot",
    }


def test_changing_oracle_cannot_change_runtime_result():
    profile = build_runtime_event_profiles(20)[10]
    altered = deepcopy(profile)
    altered["oracle"] = {
        "expected_outcome": "human_escalation",
        "expected_reason": "FabricatedOracleLabel",
    }

    original = runtime_latency.local_repair(case(), profile["event"])
    changed_oracle = runtime_latency.local_repair(case(), altered["event"])

    assert profile["event"] == altered["event"]
    assert original == changed_oracle


def test_boundary_event_returns_structured_escalation_without_effect():
    event = build_runtime_event(0, "uav_fault", replacement_available=False)

    observed = runtime_latency.local_repair(case(), event)

    assert observed["status"] == "HumanEscalated"
    assert observed["structured_failure"] == {
        "kind": "StructuredFailure",
        "reason": "NoReplacementUAV",
    }
    assert "external_effect" not in observed


def test_boundary_metric_scores_observed_runtime_state(monkeypatch):
    real_local_repair = runtime_latency.local_repair

    def misroute_boundary(value, event):
        observed = real_local_repair(value, event)
        if observed["status"] == "HumanEscalated":
            observed["status"] = "Recovered"
            observed.pop("structured_failure")
        return observed

    monkeypatch.setattr(runtime_latency, "local_repair", misroute_boundary)

    results = evaluate_runtime([case(case_id=f"I{index:04d}") for index in range(20)])

    assert results["skyrescue"]["boundary_handling_accuracy"] == 0.95
    assert results["skyrescue"]["unrecoverable_handling_accuracy"] == 0.0


def test_unrecoverable_cases_are_escalated_not_counted_as_repairs():
    cases = [case(case_id=f"I{index:04d}") for index in range(1, 173)]
    results = evaluate_runtime(cases)
    assert results["skyrescue"]["repair_success_rate"] == 1.0
    assert results["skyrescue"]["unrecoverable_handling_accuracy"] == 1.0
    assert results["skyrescue"]["boundary_handling_accuracy"] == 1.0
    assert results["skyrescue"]["human_escalation_rate"] == 0.0465
    assert results["direct_action"]["workflow_change_ratio"] is None
    assert results["static_dag"]["commitment_preservation_rate"] is None


def test_event_anchor_changes_closure_for_the_same_event_type():
    near = build_runtime_event(
        0,
        "new_task",
        directly_affected_nodes=["RepairOrEscalate"],
    )
    upstream = build_runtime_event(
        0,
        "new_task",
        directly_affected_nodes=["CommitMission:1"],
    )

    near_result = runtime_latency.local_repair(case(), near)
    upstream_result = runtime_latency.local_repair(case(), upstream)

    assert near_result["impact_closure"] == ["RepairOrEscalate"]
    assert upstream_result["impact_closure"] == [
        "CommitMission:1",
        "MonitorExecution",
        "Compensate",
        "RepairOrEscalate",
    ]
    assert near_result["change_ratio"] == 1 / 11
    assert upstream_result["change_ratio"] == 4 / 11


def test_runtime_summary_reports_observed_closure_statistics():
    results = evaluate_runtime([case()])

    assert results["skyrescue"]["impact_closure_mean_nodes"] == 8
    assert results["skyrescue"]["impact_closure_p95_nodes"] == 8
    assert results["skyrescue"]["impact_closure_min_nodes"] == 8
    assert results["skyrescue"]["impact_closure_max_nodes"] == 8
    assert results["full_replan"]["impact_closure_mean_nodes"] == 8


def test_changing_a_dependency_edge_changes_traversed_closure():
    state = runtime_latency.build_initial_state(case())
    event = build_runtime_event(
        0,
        "uav_fault",
        directly_affected_nodes=["MatchResource:1"],
    )
    original = runtime_latency._impact_closure(event, state)

    state["successors"]["ReserveAirspace:1"] = []
    rewired = runtime_latency._impact_closure(event, state)

    assert original == [
        "MatchResource:1",
        "ReserveAirspace:1",
        "SafetyPrecheck:1",
        "CommitMission:1",
        "MonitorExecution",
        "Compensate",
        "RepairOrEscalate",
    ]
    assert rewired == ["MatchResource:1", "ReserveAirspace:1"]


def test_unknown_event_anchor_fails_closed_without_an_external_effect():
    before = runtime_latency.build_initial_state(case())
    event = build_runtime_event(
        0,
        "uav_fault",
        directly_affected_nodes=["UnknownWorkflowNode"],
    )

    observed = runtime_latency.local_repair(case(), event)

    assert observed["status"] == "HumanEscalated"
    assert observed["structured_failure"] == {
        "kind": "StructuredFailure",
        "reason": "UnknownAffectedNode",
    }
    assert observed["bindings"] == before["bindings"]
    assert observed["receipts"] == before["receipts"]
    assert "external_effect" not in observed


def test_default_event_anchors_exist_and_graph_encodes_dependency_kinds():
    valid_cases = [
        case(),
        case(
            instruction=(
                "优先向东南片区孤岛运送急救药品，"
                "并建立通信中继持续30分钟。"
            ),
        ),
    ]
    for value in valid_cases:
        state = runtime_latency.build_initial_state(value)
        for index, event_type in enumerate(RUNTIME_EVENT_TYPES):
            event = build_runtime_event(index, event_type)
            assert set(event["directly_affected_nodes"]) <= set(state["nodes"])
        assert set(state["successors"]) == set(state["nodes"])
        assert runtime_latency._impact_closure(
            build_runtime_event(
                0,
                "new_task",
                directly_affected_nodes=["ParseIntent"],
            ),
            state,
        ) == state["nodes"]

        assert {edge["kind"] for edge in state["dependency_edges"]} == {
            "task",
            "resource",
            "state",
            "reservation",
        }


def test_directly_affected_nodes_are_type_checked_as_observable_input():
    event = build_runtime_event(0, "new_task")
    event["directly_affected_nodes"] = "DiscoverSkills"

    assert adjudicate_runtime_event(event) == {
        "kind": "StructuredFailure",
        "control": "human_escalation",
        "reason": "InvalidRuntimeEvent",
    }


def test_runtime_commitment_metric_observes_before_after_binding(monkeypatch):
    real_local_repair = runtime_latency.local_repair

    def repair_with_misreported_closure(value, profile):
        repaired = real_local_repair(value, profile)
        # The real new-task path changed CommitMission:1.  Declaring only the
        # final control node as the closure makes that commitment protected;
        # the observed binding/reservation delta must therefore score zero.
        repaired["impact_closure"] = [repaired["nodes"][-1]]
        return repaired

    monkeypatch.setattr(runtime_latency, "local_repair", repair_with_misreported_closure)

    results = evaluate_runtime([case()])

    assert results["skyrescue"]["commitment_preservation_rate"] == 0.0

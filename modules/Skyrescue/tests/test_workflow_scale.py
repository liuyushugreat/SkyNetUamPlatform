"""Regression tests for the single-graph workflow scale experiment."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from scripts import run_workflow_scale as scale


def frozen_cases() -> list[dict[str, object]]:
    return [
        {
            "case_id": "I0001",
            "expected_failure": None,
            "expected_tasks": [
                {
                    "task_type": "MedicalDelivery",
                    "target_zone": "Zone-SE-07",
                    "priority": "Critical",
                    "skill": "medical_payload",
                },
                {
                    "task_type": "Search",
                    "target_zone": "Zone-NW-03",
                    "priority": "Critical",
                    "skill": "camera",
                },
            ],
        }
    ]


def compiled(size: int = 12) -> scale.CompiledWorkflow:
    candidate = scale.generate_typed_candidate(size=size, seed=20260905)
    return scale.compile_typed_candidate(candidate)


def test_size_is_the_number_of_tasks_in_one_graph() -> None:
    small = compiled(9)
    large = compiled(37)

    assert len(small.tasks) == len(small.planned_state.tasks) == 9
    assert len(large.tasks) == len(large.planned_state.tasks) == 37
    assert small.workflow_id != large.workflow_id
    assert small.topological_order[0] == "task-00000"
    assert large.topological_order[-1] == "task-00036"


def test_compiler_emits_only_a_plan_and_runtime_fixture_adds_receipts() -> None:
    workflow = compiled(11)

    assert workflow.planned_state.version == 0
    assert all(state.phase == "Planned" for state in workflow.planned_state.tasks.values())
    assert all(state.reservation.startswith("planned-reservation:") for state in workflow.planned_state.tasks.values())
    assert all(state.receipt is None for state in workflow.planned_state.tasks.values())

    executed = scale.build_committed_runtime_fixture(workflow)
    assert executed is not workflow.planned_state
    assert executed.version == 1
    assert all(state.phase == "Committed" for state in executed.tasks.values())
    assert all(state.receipt and state.receipt.startswith("fixture-receipt:") for state in executed.tasks.values())
    assert all(state.receipt is None for state in workflow.planned_state.tasks.values())


def test_compile_validates_each_task_instead_of_accepting_declared_size() -> None:
    candidate = scale.generate_typed_candidate(size=10, seed=7)
    malformed = replace(candidate.tasks[6], skill="unregistered_skill")
    bad_candidate = replace(candidate, tasks=(*candidate.tasks[:6], malformed, *candidate.tasks[7:]))

    with pytest.raises(ValueError, match="UnknownSkill"):
        scale.compile_typed_candidate(bad_candidate)


def test_causal_closure_is_derived_from_graph_edges_not_a_fixed_size() -> None:
    workflow = compiled(12)
    affected = frozenset({"task-00004"})
    ordinary = scale.causal_impact_closure(workflow, affected)
    assert ordinary == {"task-00004", "task-00011"}

    # Add a real causal path and observe the closure grow. A hard-coded
    # event-to-size table would leave the result unchanged.
    successors = dict(workflow.successors)
    successors["task-00004"] = (*successors["task-00004"], "task-00005")
    rewired = replace(workflow, successors=successors)
    expanded = scale.causal_impact_closure(rewired, affected)
    assert expanded == {"task-00004", "task-00005", "task-00011"}


def test_change_ratio_is_computed_from_executed_before_after_state() -> None:
    workflow = compiled(20)
    before = scale.build_committed_runtime_fixture(workflow)
    after = deepcopy(before)
    event = scale.RuntimeEvent(
        event_id="observed-delta",
        event_type="uav_fault",
        directly_affected=frozenset({"task-00008"}),
    )

    observed = scale.execute_local_repair(workflow, before, after, event)

    closure = set(observed.impact_closure)
    assert observed.changed_nodes == len(observed.impact_closure) == 2
    assert observed.total_nodes == 20
    assert observed.change_ratio == pytest.approx(2 / 20)
    assert observed.preserved_commitments == observed.protected_commitments == 18
    assert observed.commitment_preservation_rate == 1.0
    assert all(
        after.tasks[task_id].phase == "Recovered"
        and after.tasks[task_id].reservation
        and after.tasks[task_id].receipt
        for task_id in closure
    )
    assert all(
        after.tasks[task_id].phase == "Committed"
        and after.tasks[task_id] == before.tasks[task_id]
        for task_id in set(workflow.tasks) - closure
    )


def test_closure_external_tampering_is_detected() -> None:
    workflow = compiled(16)
    before = scale.build_committed_runtime_fixture(workflow)
    after = deepcopy(before)
    event = scale.RuntimeEvent(
        event_id="tamper-test",
        event_type="communication_loss",
        directly_affected=frozenset({"task-00007"}),
    )

    def corrupt_external_commitment(
        state: scale.WorkflowRuntimeState,
        closure: frozenset[str],
    ) -> None:
        outside = next(task_id for task_id in workflow.topological_order if task_id not in closure)
        state.tasks[outside].receipt = "forged-receipt"

    with pytest.raises(ValueError, match="closure-external commitment changed"):
        scale.execute_local_repair(
            workflow,
            before,
            after,
            event,
            before_recheck=corrupt_external_commitment,
        )


def test_worker_retains_constant_repeat_rows_not_one_row_per_task(tmp_path) -> None:
    dataset = tmp_path / "intent"
    dataset.mkdir()
    rows = [
        {
            "case_id": "I0001",
            "expected_failure": None,
            "expected_tasks": frozen_cases()[0]["expected_tasks"],
        }
    ]
    (dataset / "intent_cases.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    small = scale.run_one(dataset, size=8, seed=11, warmup_rounds=1, repeats=3)
    large = scale.run_one(dataset, size=80, seed=11, warmup_rounds=1, repeats=3)

    assert len(small["events"]) == len(large["events"]) == 3
    assert small["run"]["task_graph_size"] == 8
    assert large["run"]["task_graph_size"] == 80
    assert all(row["total_nodes"] == 80 for row in large["events"])

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from typing import Any, Callable

import pytest

from skyrescue import langgraph_baseline, runtime_latency
from skyrescue.core_contract import IdempotentReceiver
from skyrescue.workflow import build_runtime_event


def case(case_id: str = "I0001") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "instruction": "优先向东南片区孤岛运送急救药品，15分钟内完成。",
        "expected_tasks": [
            {
                "task_type": "MedicalDelivery",
                "target_zone": "Zone-SE-07",
                "priority": "Critical",
                "skill": "medical_payload",
                "deadline_s": 900,
            }
        ],
        "expected_failure": None,
        "requires_human_approval": False,
        "approval_granted": False,
        "conditional": False,
    }


class SynchronousGraph:
    """Test double for orchestration; all business semantics remain native."""

    def __init__(
        self,
        receiver: IdempotentReceiver,
        *,
        repair_invocations: int = 1,
        mutate_after: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.receiver = receiver
        self.repair_invocations = repair_invocations
        self.mutate_after = mutate_after

    def invoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        assert config["configurable"]["thread_id"]
        state = dict(payload)
        state.update(langgraph_baseline._prepare(state))
        state.update(langgraph_baseline._adjudicate(state))
        if langgraph_baseline._route_after_prepare(state) == "repair":
            for _ in range(self.repair_invocations):
                state.update(langgraph_baseline._repair(state, self.receiver))
            if self.mutate_after is not None:
                self.mutate_after(state["after_state"])
        else:
            state.update(langgraph_baseline._human_escalation(state))
        return state


def graph_factory(
    *,
    repair_invocations: int = 1,
    mutate_after: Callable[[dict[str, Any]], None] | None = None,
):
    def build(
        connection: sqlite3.Connection,
        receiver: IdempotentReceiver,
    ) -> SynchronousGraph:
        connection.execute("CREATE TABLE semantic_test_checkpoint (id INTEGER)")
        connection.commit()
        return SynchronousGraph(
            receiver,
            repair_invocations=repair_invocations,
            mutate_after=mutate_after,
        )

    return build


def evaluate(
    tmp_path: Path,
    *,
    repair_invocations: int = 1,
    mutate_after: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return langgraph_baseline.evaluate_langgraph_runtime(
        [case()],
        tmp_path / "langgraph.sqlite",
        graph_factory=graph_factory(
            repair_invocations=repair_invocations,
            mutate_after=mutate_after,
        ),
    )


def test_langgraph_embedding_calls_native_repair_and_receiver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_repair = runtime_latency.local_repair
    calls = 0

    def observed_repair(value, profile, receiver=None):
        nonlocal calls
        calls += 1
        return real_repair(value, profile, receiver=receiver)

    monkeypatch.setattr(runtime_latency, "local_repair", observed_repair)
    metrics = evaluate(tmp_path)

    assert calls == 1
    assert metrics["repair_success_rate"] == 1.0
    assert metrics["external_invocations"] == 1
    assert metrics["external_effects"] == 1
    assert metrics["stored_repair_receipts"] == 1
    assert metrics["receipt_completion_rate"] == 1.0
    assert metrics["impact_closure_mean_nodes"] == 8
    assert metrics["impact_closure_p95_nodes"] == 8
    assert metrics["impact_closure_min_nodes"] == 8
    assert metrics["impact_closure_max_nodes"] == 8


def test_change_ratio_follows_executed_closure_not_old_constant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_runtime_event(
        0,
        "new_task",
        directly_affected_nodes=["RepairOrEscalate"],
    )
    monkeypatch.setattr(
        langgraph_baseline,
        "build_runtime_event_profiles",
        lambda _count: [
            {
                "event": event,
                "oracle": {"expected_outcome": "repair", "expected_reason": None},
            }
        ],
    )

    metrics = evaluate(tmp_path)
    native = runtime_latency.local_repair(case(), event)

    assert metrics["workflow_change_ratio"] == round(native["change_ratio"], 4) == 0.0909
    assert metrics["commitment_preservation_rate"] == 1.0


def test_commitment_metric_observes_binding_and_receipt_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_runtime_event(
        0,
        "new_task",
        directly_affected_nodes=["RepairOrEscalate"],
    )
    monkeypatch.setattr(
        langgraph_baseline,
        "build_runtime_event_profiles",
        lambda _count: [
            {
                "event": event,
                "oracle": {"expected_outcome": "repair", "expected_reason": None},
            }
        ],
    )

    def corrupt_protected_commitment(after: dict[str, Any]) -> None:
        after["bindings"]["CommitMission:1"] = "U9999"
        after["receipts"]["CommitMission:1"] = "tampered-receipt"

    metrics = evaluate(tmp_path, mutate_after=corrupt_protected_commitment)

    assert metrics["commitment_preservation_rate"] == 0.0


def test_duplicate_calls_come_from_receiver_counts(tmp_path: Path) -> None:
    metrics = evaluate(tmp_path, repair_invocations=2)

    assert metrics["duplicate_external_calls"] == 1
    assert metrics["duplicate_external_effects"] == 0
    assert metrics["external_invocations"] == 2
    assert metrics["external_effects"] == 1


def test_missing_persisted_receipt_fails_repair_metric(tmp_path: Path) -> None:
    def remove_receipt(after: dict[str, Any]) -> None:
        effect = after["external_effect"]
        after["receipts"].pop(effect["receipt_slot"])

    metrics = evaluate(tmp_path, mutate_after=remove_receipt)

    assert metrics["repair_success_rate"] == 0.0
    assert metrics["receipt_completion_rate"] == 0.0


def test_no_legacy_metric_constants_remain_in_evaluator() -> None:
    source = inspect.getsource(langgraph_baseline.evaluate_langgraph_runtime)
    assert "committed_nodes" not in source
    assert "_impact_closure_size" not in source
    assert '"receipt_check"' not in source
    assert "duplicate_external_calls\", 0" not in source
    assert not hasattr(runtime_latency, "_CLOSURE_SIZE")
    closure_source = inspect.getsource(runtime_latency._impact_closure)
    assert "successors" in closure_source
    assert "event_type" not in closure_source


def test_deterministically_adjudicated_boundary_and_receipt_evidence_are_scoped(
    tmp_path: Path,
) -> None:
    cases = [case(f"I{index:04d}") for index in range(1, 21)]
    metrics = langgraph_baseline.evaluate_langgraph_runtime(
        cases,
        tmp_path / "boundary-naming.sqlite",
        graph_factory=graph_factory(),
    )

    assert "unrecoverable_handling_accuracy" not in metrics
    assert "workload_routed_escalation_rate" not in metrics
    assert metrics["boundary_handling_accuracy"] == 1.0
    assert metrics["structured_failure_accuracy"] == 1.0
    assert "evidence_completeness" not in metrics
    assert metrics["receipt_evidence_completeness"] == 1.0
    assert "not ML classification" in metrics["comparison_boundary"]


def test_graph_decision_is_unchanged_when_evaluator_oracle_is_relabelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_runtime_event(0, "new_task", source_trusted=False)
    deliberately_wrong_oracle = {
        "event": event,
        "oracle": {"expected_outcome": "repair", "expected_reason": None},
    }
    monkeypatch.setattr(
        langgraph_baseline,
        "build_runtime_event_profiles",
        lambda _count: [deliberately_wrong_oracle],
    )

    metrics = langgraph_baseline.evaluate_langgraph_runtime(
        [case()],
        tmp_path / "oracle-separation.sqlite",
        graph_factory=graph_factory(),
    )

    assert metrics["human_escalations"] == 1
    assert metrics["boundary_handling_accuracy"] == 0.0
    assert metrics["external_invocations"] == 0


def test_production_builder_retains_stategraph_retry_and_sqlite_path() -> None:
    graph_source = inspect.getsource(langgraph_baseline._build_graph)
    evaluator_source = inspect.getsource(langgraph_baseline.evaluate_langgraph_runtime)
    assert "StateGraph" in graph_source
    assert "RetryPolicy" in graph_source
    assert "checkpointer=checkpointer" in graph_source
    assert "SqliteSaver" in evaluator_source


def test_real_stategraph_sqlite_path_when_dependency_is_available(tmp_path: Path) -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("langgraph.checkpoint.sqlite")

    metrics = langgraph_baseline.evaluate_langgraph_runtime(
        [case()],
        tmp_path / "real-langgraph.sqlite",
    )

    assert metrics["repair_success_rate"] == 1.0
    assert metrics["external_invocations"] == 1
    assert metrics["external_effects"] == 1

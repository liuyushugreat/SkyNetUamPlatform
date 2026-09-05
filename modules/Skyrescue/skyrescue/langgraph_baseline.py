"""LangGraph embedding of the native SkyRescue repair and commit semantics.

LangGraph supplies StateGraph control flow, retry policy, and SQLite
checkpointing.  Its repair node calls the exact native ``local_repair`` path,
including the shared idempotent receiver.  Metrics are recomputed from observed
before/after state and receiver counters, never from graph-authored constants.
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

from . import runtime_latency
from .core_contract import IdempotentReceiver
from .workflow import build_runtime_event_profiles, compile_case


class RuntimeState(TypedDict, total=False):
    case: dict[str, Any]
    event: dict[str, Any]
    adjudication: dict[str, Any]
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    repaired: bool
    escalated: bool
    failure: str | None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _prepare(state: RuntimeState) -> RuntimeState:
    """Materialize the same pre-event business state used by native repair."""

    return {"before_state": runtime_latency.build_initial_state(state["case"])}


def _adjudicate(state: RuntimeState) -> RuntimeState:
    """Apply the shared deterministic gate to observable fields only."""

    return {
        "adjudication": runtime_latency.adjudicate_runtime_event(state["event"]),
    }


def _route_after_prepare(state: RuntimeState) -> str:
    return state["adjudication"]["control"]


def _repair(state: RuntimeState, receiver: IdempotentReceiver) -> RuntimeState:
    """Invoke the native repair-and-commit implementation without duplication."""

    after = runtime_latency.local_repair(
        state["case"],
        state["event"],
        receiver=receiver,
    )
    return {
        "after_state": after,
        "repaired": after.get("status") == "Recovered",
        "escalated": False,
        "failure": None,
    }


def _human_escalation(state: RuntimeState) -> RuntimeState:
    after = runtime_latency.human_escalation_state(
        deepcopy(state["before_state"]),
        state["adjudication"],
    )
    return {
        "after_state": after,
        "repaired": False,
        "escalated": True,
        "failure": after["structured_failure"]["reason"],
    }


def _build_graph(checkpointer: Any, receiver: IdempotentReceiver):
    """Compile the production LangGraph/SQLite embedding lazily."""

    from langgraph.graph import END, START, StateGraph
    from langgraph.types import RetryPolicy

    def repair_node(state: RuntimeState) -> RuntimeState:
        return _repair(state, receiver)

    graph = StateGraph(RuntimeState)
    graph.add_node("prepare", _prepare)
    graph.add_node("adjudicate", _adjudicate)
    graph.add_node(
        "repair",
        repair_node,
        retry_policy=RetryPolicy(max_attempts=2, jitter=False),
    )
    graph.add_node("human_escalation", _human_escalation)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "adjudicate")
    graph.add_conditional_edges(
        "adjudicate",
        _route_after_prepare,
        {"repair": "repair", "human_escalation": "human_escalation"},
    )
    graph.add_edge("repair", END)
    graph.add_edge("human_escalation", END)
    return graph.compile(checkpointer=checkpointer)


def _receipt_is_observed(
    after: dict[str, Any],
    receiver: IdempotentReceiver,
) -> bool:
    """Validate a persisted receipt against the receiver's observed counters."""

    effect = after.get("external_effect")
    if not isinstance(effect, dict):
        return False
    key = effect.get("idempotency_key")
    slot = effect.get("receipt_slot")
    if not isinstance(key, str) or not isinstance(slot, str):
        return False
    receiver_receipt = receiver.query(key)
    return bool(
        receiver_receipt is not None
        and after.get("receipts", {}).get(slot) == receiver_receipt.receipt_id
        and effect.get("receipt_id") == receiver_receipt.receipt_id
        and effect.get("receipt_count") == 1
        and effect.get("invoke_count") == receiver.invoke_count(key)
        and effect.get("effect_count") == receiver.effect_count(key) == 1
    )


def _remove_sqlite_files(database: Path) -> None:
    """Remove a previous SQLite checkpoint and any journal sidecars."""

    for candidate in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
        if candidate.exists():
            candidate.unlink()


def evaluate_langgraph_runtime(
    cases: list[dict[str, Any]],
    checkpoint_path: Path,
    *,
    graph_factory: Callable[[sqlite3.Connection, IdempotentReceiver], Any] | None = None,
) -> dict[str, Any]:
    """Run native repair/commit semantics inside a persisted StateGraph."""

    valid = [case for case in cases if case.get("expected_failure") is None]
    if not valid:
        raise ValueError("The LangGraph runtime benchmark requires executable cases")
    profiles = build_runtime_event_profiles(len(valid))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_sqlite_files(checkpoint_path)

    receiver = IdempotentReceiver()
    latencies: list[float] = []
    repaired = actual_escalations = changed = node_total = 0
    protected = preserved = receipt_complete = 0
    boundary_correct = structured_failure_correct = 0
    impact_closure_sizes: list[int] = []
    # Production evaluation always takes this branch.  ``graph_factory`` is a
    # narrow dependency-injection seam for environments without LangGraph.
    with sqlite3.connect(checkpoint_path, check_same_thread=False) as conn:
        if graph_factory is None:
            from langgraph.checkpoint.sqlite import SqliteSaver

            app = _build_graph(SqliteSaver(conn), receiver)
        else:
            app = graph_factory(conn, receiver)
        for case, profile in zip(valid, profiles):
            event = profile["event"]
            oracle = profile["oracle"]
            compiled = compile_case(case, "skyrescue")
            if not compiled.executable:
                raise ValueError(f"Shared compiler unexpectedly rejected {case['case_id']}")
            started = time.perf_counter()
            final_state = app.invoke(
                {"case": case, "event": event},
                {"configurable": {"thread_id": f"langgraph-{case['case_id']}"}},
            )
            latencies.append((time.perf_counter() - started) * 1000)
            before = final_state.get("before_state")
            after = final_state.get("after_state")
            if not isinstance(before, dict) or not isinstance(after, dict):
                raise ValueError("LangGraph did not return observable before/after state")

            observed_outcome = (
                "human_escalation"
                if final_state.get("escalated") is True
                else "repair"
            )
            observed_reason = final_state.get("failure")
            expected_outcome = oracle["expected_outcome"]
            expected_reason = oracle["expected_reason"]
            exact_boundary_match = (
                observed_outcome == expected_outcome
                and observed_reason == expected_reason
            )
            boundary_correct += int(exact_boundary_match)
            actual_escalations += int(observed_outcome == "human_escalation")

            if expected_outcome == "repair":
                if observed_outcome != "repair":
                    continue
                closure = after.get("impact_closure")
                if not isinstance(closure, list):
                    raise ValueError("Native repair did not return an impact closure")
                measured = runtime_latency.compare_repair_states(before, after, closure)
                receipt_observed = _receipt_is_observed(after, receiver)
                repaired += int(after.get("status") == "Recovered" and receipt_observed)
                receipt_complete += int(receipt_observed)
                changed += int(measured["changed_nodes"])
                node_total += int(measured["total_nodes"])
                protected += int(measured["protected_commitments"])
                preserved += int(measured["preserved_commitments"])
                impact_closure_sizes.append(len(closure))
            else:
                unchanged = (
                    before.get("bindings") == after.get("bindings")
                    and before.get("receipts") == after.get("receipts")
                )
                structured_failure_correct += int(
                    exact_boundary_match
                    and unchanged
                    and after.get("status") == "HumanEscalated"
                    and after.get("structured_failure", {}).get("reason")
                    == expected_reason
                )

    recoverable = sum(
        profile["oracle"]["expected_outcome"] == "repair" for profile in profiles
    )
    unrecoverable = len(profiles) - recoverable
    return {
        "implementation": (
            "LangGraph StateGraph + SQLite checkpointer embedding native "
            "runtime_latency.local_repair and IdempotentReceiver"
        ),
        "workflows": len(valid),
        "recoverable_events": recoverable,
        "unrecoverable_events": unrecoverable,
        "repair_success_rate": round(repaired / recoverable, 4) if recoverable else None,
        "boundary_handling_accuracy": round(boundary_correct / len(profiles), 4),
        "structured_failure_accuracy": (
            round(structured_failure_correct / unrecoverable, 4)
            if unrecoverable
            else None
        ),
        "workflow_change_ratio": round(changed / node_total, 4) if node_total else None,
        "commitment_preservation_rate": (
            round(preserved / protected, 4) if protected else None
        ),
        "impact_closure_mean_nodes": (
            round(sum(impact_closure_sizes) / len(impact_closure_sizes), 4)
            if impact_closure_sizes
            else None
        ),
        "impact_closure_p95_nodes": (
            _percentile([float(value) for value in impact_closure_sizes], 0.95)
            if impact_closure_sizes
            else None
        ),
        "impact_closure_min_nodes": min(impact_closure_sizes) if impact_closure_sizes else None,
        "impact_closure_max_nodes": max(impact_closure_sizes) if impact_closure_sizes else None,
        "duplicate_external_calls": receiver.duplicate_invocations,
        "duplicate_external_effects": receiver.duplicate_effects,
        "external_invocations": receiver.total_invocations,
        "external_effects": receiver.total_effects,
        "stored_repair_receipts": receipt_complete,
        "receipt_completion_rate": (
            round(receipt_complete / recoverable, 4) if recoverable else None
        ),
        "receipt_evidence_completeness": (
            round(receipt_complete / recoverable, 4) if recoverable else None
        ),
        "repair_p50_ms": round(_percentile(latencies, 0.50), 4),
        "repair_p95_ms": round(_percentile(latencies, 0.95), 4),
        "repair_p99_ms": round(_percentile(latencies, 0.99), 4),
        "human_escalations": actual_escalations,
        "metric_provenance": {
            "workflow_change_ratio": "runtime_latency.compare_repair_states(before, after, closure)",
            "commitment_preservation_rate": (
                "committed closure-external binding/reservation/receipt snapshots"
            ),
            "duplicate_external_calls": "IdempotentReceiver invocation counters",
            "receipt_evidence_completeness": (
                "persisted repair receipt matched to the receiver receipt and counters"
            ),
            "boundary_handling_accuracy": (
                "observed graph control and structured reason compared outside the graph "
                "with the separately stored benchmark oracle"
            ),
            "structured_failure_accuracy": (
                "exact structured reason plus unchanged binding/receipt state on the eight "
                "oracle boundary cases; this deterministic gate is not an ML classifier"
            ),
        },
        "comparison_boundary": (
            "The typed compiler, task inputs, observable events, native repair function, and "
            "idempotent receiver are shared. LangGraph contributes orchestration, retry, and "
            "SQLite graph-state checkpointing. The receiver is process-local and is not "
            "crash-persisted by that checkpointer. The graph receives no expected outcome or "
            "failure label: a deterministic safety gate maps observable event fields to repair "
            "or structured escalation, and the evaluator alone compares that result with the "
            "separate oracle. This is mechanism conformance, not ML classification, and these "
            "results are not native LangGraph guarantees."
        ),
    }

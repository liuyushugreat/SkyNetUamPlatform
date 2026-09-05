#!/usr/bin/env python3
"""Run deterministic DevOps and UAV workloads through one runtime contract.

Every reported count and rate is derived from executed contract objects.  The
experiment makes no LLM calls, records no wall-clock measurements, and embeds
no precomputed result table.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import skyrescue.core_contract as core_contract_module  # noqa: E402
from skyrescue.core_contract import (  # noqa: E402
    CORE_CONTRACT_COMPONENTS,
    AdmissionResult,
    ContractAdapter,
    ContractState,
    EventResult,
    RuntimeContract,
    RuntimeEvent,
    Workflow,
)
from skyrescue.devops_adapter import (  # noqa: E402
    DEFAULT_DEVOPS_SEED,
    DEVOPS_ADAPTER_REPLACEMENTS,
    DevOpsAdapter,
    build_devops_events,
    build_devops_instruction_cases,
)
from skyrescue.uav_contract_adapter import (  # noqa: E402
    DEFAULT_UAV_CONTRACT_SEED,
    UAV_ADAPTER_REPLACEMENTS,
    UAV_SOURCE_RUNTIME_EVENT_TYPES,
    UAVEmergencyAdapter,
    build_uav_events,
    build_uav_instruction_cases,
)

LabeledRuntimeEvent = tuple[RuntimeEvent, str, str | None]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_fingerprint(
    cases: Sequence[dict[str, object]],
    events: Sequence[LabeledRuntimeEvent],
) -> str:
    event_records = [
        {
            "event_id": event.event_id,
            "workflow_id": event.workflow_id,
            "event_type": event.event_type,
            "directly_affected": sorted(event.directly_affected),
            "expected_outcome": expected_outcome,
            "expected_failure": expected_failure,
            "metadata": dict(event.metadata),
        }
        for event, expected_outcome, expected_failure in events
    ]
    encoded = json.dumps(
        {"instruction_cases": list(cases), "events": event_records},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _label_events(
    events: Sequence[RuntimeEvent],
    oracle: Callable[[RuntimeEvent], tuple[str, str | None]],
) -> list[LabeledRuntimeEvent]:
    return [(event, *oracle(event)) for event in events]


def _devops_oracle(event: RuntimeEvent) -> tuple[str, str | None]:
    if event.event_type == "duplicate_event":
        return "duplicate_ignored", None
    failures = {
        "permission_change": "PermissionDenied",
        "approval_timeout": "HumanApprovalRequired",
    }
    failure = failures.get(event.event_type)
    return ("escalate", failure) if failure is not None else ("repair", None)


def _uav_oracle(event: RuntimeEvent) -> tuple[str, str | None]:
    if event.event_type == "duplicate_event":
        return "duplicate_ignored", None
    if event.event_type == "priority_preemption" and event.metadata.get(
        "approval_timed_out"
    ):
        return "escalate", "HumanApprovalTimeout"
    if event.event_type == "danger_zone" and event.metadata.get("concurrent_uav_fault"):
        return "escalate", "ConcurrentFaultAndDangerZone"
    return "repair", None


CORE_METHOD_NAMES = (
    "admit",
    "impact_closure",
    "commit_external_effect",
    "process_event",
)


def _execute_domain_untraced(
    *,
    adapter: ContractAdapter,
    cases: list[dict[str, object]],
    event_builder: Callable[[Sequence[Workflow]], list[LabeledRuntimeEvent]],
    event_count: int,
    seed: int,
    effect_node: str,
    initial_effect_kind: str,
) -> tuple[dict[str, Any], RuntimeContract]:
    """Execute one adapter without changing the shared contract algorithm."""

    contract = RuntimeContract(adapter)
    admission_rows: list[tuple[dict[str, object], AdmissionResult]] = []
    workflows: dict[str, Workflow] = {}
    admission_correct = 0
    structured_failure_correct = 0
    failure_counts: Counter[str] = Counter()

    for case in cases:
        case_id = str(case["case_id"])
        result = contract.admit(case_id, str(case["instruction"]))
        expected_executable = bool(case["expected_executable"])
        expected_failure = case["expected_failure"]
        observed_failure = result.failure.code if result.failure is not None else None
        exact_match = (
            result.executable == expected_executable
            and (expected_executable or observed_failure == expected_failure)
        )
        admission_correct += int(exact_match)
        if not expected_executable:
            structured_failure_correct += int(observed_failure == expected_failure)
        if observed_failure is not None:
            failure_counts[observed_failure] += 1
        if result.workflow is not None:
            workflows[result.workflow.workflow_id] = result.workflow
        admission_rows.append((case, result))

    instruction_count = len(cases)
    expected_executable_count = sum(bool(case["expected_executable"]) for case in cases)
    admitted_count = len(workflows)
    negative_count = instruction_count - expected_executable_count
    if admission_correct != instruction_count:
        raise RuntimeError(
            f"{adapter.domain_name} admission diverged from the deterministic case oracle"
        )

    initial_commits = []
    for case, result in admission_rows:
        if result.workflow is None:
            continue
        initial_commit = contract.commit_external_effect(
            result.workflow,
            idempotency_key=f"{result.workflow.workflow_id}:initial:v1",
            causal_parent=str(case["case_id"]),
            payload={"kind": initial_effect_kind},
            node_id=effect_node,
        )
        if initial_commit.state != ContractState.COMMITTED:
            raise RuntimeError(
                f"An admitted {adapter.domain_name} action did not reach Committed"
            )
        initial_commits.append(initial_commit)

    events = event_builder(list(workflows.values()))
    if len(events) != event_count:
        raise RuntimeError("The event builder returned an unexpected number of requests")
    event_rows: list[tuple[RuntimeEvent, str, str | None, EventResult]] = []
    boundary_correct = 0
    event_status_counts: Counter[str] = Counter()
    event_class_counts: Counter[str] = Counter()
    event_class_status: dict[str, Counter[str]] = defaultdict(Counter)

    for event, expected_outcome, expected_failure in events:
        result = contract.process_event(workflows[event.workflow_id], event)
        expected_status = {
            "repair": "repaired",
            "escalate": "escalated",
            "duplicate_ignored": "duplicate_ignored",
        }[expected_outcome]
        exact_match = result.status == expected_status
        if expected_outcome == "escalate":
            exact_match = exact_match and result.failure is not None
            exact_match = exact_match and result.failure.code == expected_failure
        boundary_correct += int(exact_match)
        event_status_counts[result.status] += 1
        event_class_counts[event.event_type] += 1
        event_class_status[event.event_type][result.status] += 1
        event_rows.append((event, expected_outcome, expected_failure, result))

    if boundary_correct != event_count:
        raise RuntimeError(
            f"{adapter.domain_name} runtime diverged from the deterministic event oracle"
        )

    repair_rows = [
        (event, result)
        for event, expected_outcome, _, result in event_rows
        if expected_outcome == "repair"
    ]
    escalation_rows = [
        (event, result)
        for event, expected_outcome, _, result in event_rows
        if expected_outcome == "escalate"
    ]
    replay_rows = [
        (event, result)
        for event, expected_outcome, _, result in event_rows
        if expected_outcome == "duplicate_ignored"
    ]
    repaired_count = sum(result.status == "repaired" for _, result in repair_rows)
    escalation_correct = sum(
        result.status == "escalated"
        and result.failure is not None
        and result.failure.code == expected_failure
        for event, expected_outcome, expected_failure, result in event_rows
        if expected_outcome == "escalate"
    )
    replay_correct = sum(result.status == "duplicate_ignored" for _, result in replay_rows)
    changed_nodes = sum(result.changed_nodes for _, result in repair_rows)
    repair_nodes = sum(result.total_nodes for _, result in repair_rows)
    protected_commitments = sum(result.protected_commitments for _, result in repair_rows)
    preserved_commitments = sum(result.preserved_commitments for _, result in repair_rows)
    receipt_reconciliations = sum(
        result.commit_result is not None and result.commit_result.reconciled
        for _, result in repair_rows
    )
    stored_receipts = sum(len(workflow.receipts) for workflow in workflows.values())
    status_by_event_class = {
        event_type: {
            "requests": event_class_counts[event_type],
            "observed_statuses": dict(sorted(event_class_status[event_type].items())),
        }
        for event_type in sorted(event_class_counts)
    }

    return (
        {
            "domain": adapter.domain_name,
            "configuration": {
                "seed": seed,
                "instruction_requests": instruction_count,
                "event_requests": event_count,
                "input_sha256": _input_fingerprint(cases, events),
                "llm_calls": 0,
            },
            "admission": {
                "expected_executable": expected_executable_count,
                "admitted": admitted_count,
                "expected_structured_failures": negative_count,
                "observed_structured_failures": sum(failure_counts.values()),
                "compile_success_rate": _rate(admitted_count, instruction_count),
                "admission_accuracy": _rate(admission_correct, instruction_count),
                "structured_failure_accuracy": _rate(
                    structured_failure_correct,
                    negative_count,
                ),
                "failure_counts": dict(sorted(failure_counts.items())),
            },
            "runtime": {
                "event_requests": event_count,
                "unique_event_ids": len({event.event_id for event, _, _ in events}),
                "recoverable_events": len(repair_rows),
                "repaired_events": repaired_count,
                "escalation_events": len(escalation_rows),
                "correct_escalations": escalation_correct,
                "duplicate_event_requests": len(replay_rows),
                "duplicates_ignored": replay_correct,
                "repair_success_rate": _rate(repaired_count, len(repair_rows)),
                "escalation_accuracy": _rate(escalation_correct, len(escalation_rows)),
                "duplicate_event_rejection_rate": _rate(replay_correct, len(replay_rows)),
                "boundary_handling_accuracy": _rate(boundary_correct, event_count),
                "changed_nodes": changed_nodes,
                "repair_scope_nodes": repair_nodes,
                "workflow_change_ratio": _rate(changed_nodes, repair_nodes),
                "protected_commitments": protected_commitments,
                "preserved_commitments": preserved_commitments,
                "commitment_preservation_rate": _rate(
                    preserved_commitments,
                    protected_commitments,
                ),
                "external_invocations": contract.receiver.total_invocations,
                "external_effects": contract.receiver.total_effects,
                "duplicate_external_invocations": contract.receiver.duplicate_invocations,
                "duplicate_external_effects": contract.receiver.duplicate_effects,
                "stored_execution_receipts": stored_receipts,
                "receipt_reconciliations": receipt_reconciliations,
                "initial_commits": len(initial_commits),
                "observed_statuses": dict(sorted(event_status_counts.items())),
                "by_event_class": status_by_event_class,
            },
        },
        contract,
    )


def _execute_domain(
    *,
    adapter: ContractAdapter,
    cases: list[dict[str, object]],
    event_builder: Callable[[Sequence[Workflow]], list[LabeledRuntimeEvent]],
    event_count: int,
    seed: int,
    effect_node: str,
    initial_effect_kind: str,
) -> tuple[dict[str, Any], RuntimeContract]:
    """Observe calls to canonical core code objects during one domain run."""

    method_codes = {
        getattr(RuntimeContract, method_name).__code__: method_name
        for method_name in CORE_METHOD_NAMES
    }
    observed_calls: Counter[str] = Counter({name: 0 for name in CORE_METHOD_NAMES})
    previous_profiler = sys.getprofile()

    def observe_call(frame: Any, event: str, argument: Any) -> None:
        del argument
        if event == "call" and frame.f_code in method_codes:
            observed_calls[method_codes[frame.f_code]] += 1

    sys.setprofile(observe_call)
    try:
        summary, contract = _execute_domain_untraced(
            adapter=adapter,
            cases=cases,
            event_builder=event_builder,
            event_count=event_count,
            seed=seed,
            effect_node=effect_node,
            initial_effect_kind=initial_effect_kind,
        )
    finally:
        sys.setprofile(previous_profiler)

    summary["observed_core_calls"] = dict(observed_calls)
    if not all(observed_calls.values()):
        raise RuntimeError(f"{adapter.domain_name} did not execute every core method")
    return summary, contract


def run_experiment(
    *,
    seed: int = DEFAULT_DEVOPS_SEED,
    instruction_count: int = 60,
    event_count: int = 60,
    uav_seed: int = DEFAULT_UAV_CONTRACT_SEED,
    uav_instruction_count: int = 60,
    uav_event_count: int = 60,
) -> dict[str, Any]:
    """Execute both adapters and record identity of their shared core code."""

    canonical_methods = {
        name: getattr(RuntimeContract, name) for name in CORE_METHOD_NAMES
    }
    method_sha_before = {
        name: hashlib.sha256(inspect.getsource(method).encode("utf-8")).hexdigest()
        for name, method in canonical_methods.items()
    }
    resolved_core_source = inspect.getsourcefile(RuntimeContract)
    if resolved_core_source is None:
        raise RuntimeError("Cannot resolve the RuntimeContract source file")
    core_source = Path(resolved_core_source).resolve()
    module_source = Path(core_contract_module.__file__).resolve()
    if core_source != module_source:
        raise RuntimeError("RuntimeContract and its imported module resolve to different sources")
    core_sha_before = _sha256_file(core_source)

    devops_adapter = DevOpsAdapter()
    devops_summary, devops_contract = _execute_domain(
        adapter=devops_adapter,
        cases=build_devops_instruction_cases(instruction_count, seed=seed),
        event_builder=lambda workflows: _label_events(
            build_devops_events(workflows, event_count, seed=seed),
            _devops_oracle,
        ),
        event_count=event_count,
        seed=seed,
        effect_node="external_action",
        initial_effect_kind="initial_devops_action",
    )

    uav_adapter = UAVEmergencyAdapter()
    uav_summary, uav_contract = _execute_domain(
        adapter=uav_adapter,
        cases=build_uav_instruction_cases(uav_instruction_count, seed=uav_seed),
        event_builder=lambda workflows: _label_events(
            build_uav_events(workflows, uav_event_count, seed=uav_seed),
            _uav_oracle,
        ),
        event_count=uav_event_count,
        seed=uav_seed,
        effect_node="dispatch_effect",
        initial_effect_kind="initial_uav_dispatch",
    )
    uav_summary["workload_provenance"] = {
        "kind": "synthetic_template_generated",
        "reused_source_vocabularies": [
            "skyrescue.workflow.TASK_TYPES",
            "skyrescue.workflow.ZONE_ALIASES",
            "skyrescue.workflow.RUNTIME_EVENT_TYPES",
            "skyrescue.security.ACTION_PERMISSIONS",
        ],
        "source_runtime_event_types": list(UAV_SOURCE_RUNTIME_EVENT_TYPES),
        "contract_only_event_extensions": ["receipt_missing", "duplicate_event"],
        "claim_boundary": (
            "This workload tests contract reuse with UAV-domain symbols; it does not "
            "replace or validate the existing SkyRescue production path."
        ),
    }

    core_sha_after = _sha256_file(core_source)
    method_sha_after = {
        name: hashlib.sha256(
            inspect.getsource(getattr(RuntimeContract, name)).encode("utf-8")
        ).hexdigest()
        for name in CORE_METHOD_NAMES
    }
    devops_core_source = Path(inspect.getsourcefile(type(devops_contract)) or "").resolve()
    uav_core_source = Path(inspect.getsourcefile(type(uav_contract)) or "").resolve()
    devops_adapter_source = Path(
        inspect.getsourcefile(type(devops_adapter)) or ""
    ).resolve()
    uav_adapter_source = Path(inspect.getsourcefile(type(uav_adapter)) or "").resolve()
    shared_class_object = type(devops_contract) is type(uav_contract) is RuntimeContract
    shared_method_objects = all(
        getattr(type(devops_contract), name)
        is getattr(type(uav_contract), name)
        is canonical_methods[name]
        for name in CORE_METHOD_NAMES
    )
    shared_resolved_source = (
        devops_core_source == uav_core_source == core_source == module_source
    )
    runtime_event_has_oracle = "expected_outcome" in RuntimeEvent.__dataclass_fields__
    runtime_contract_reads_oracle = "expected_outcome" in inspect.getsource(
        RuntimeContract
    )
    if not all(
        (
            core_sha_before == core_sha_after,
            method_sha_before == method_sha_after,
            shared_class_object,
            shared_method_objects,
            shared_resolved_source,
            not runtime_event_has_oracle,
            not runtime_contract_reads_oracle,
        )
    ):
        raise RuntimeError("The two domain runs did not share one unchanged RuntimeContract")

    try:
        core_source_label = str(core_source.relative_to(PROJECT_ROOT))
    except ValueError:
        core_source_label = str(core_source)
    runtime_contract_name = f"{RuntimeContract.__module__}.{RuntimeContract.__qualname__}"

    return {
        "experiment": "devops_portability_with_uav_source_domain",
        "configuration": devops_summary["configuration"],
        "admission": devops_summary["admission"],
        "runtime": devops_summary["runtime"],
        "uav_source_domain": uav_summary,
        "portability_contract": {
            "reused_core_components": list(CORE_CONTRACT_COMPONENTS),
            "reused_core_component_count": len(CORE_CONTRACT_COMPONENTS),
            "replaced_adapter_components": dict(DEVOPS_ADAPTER_REPLACEMENTS),
            "replaced_adapter_component_count": len(DEVOPS_ADAPTER_REPLACEMENTS),
            "domain": devops_adapter.domain_name,
            "adapter_replacements": {
                "devops": dict(DEVOPS_ADAPTER_REPLACEMENTS),
                "uav_emergency": dict(UAV_ADAPTER_REPLACEMENTS),
            },
            "core_identity": {
                "module": RuntimeContract.__module__,
                "class": runtime_contract_name,
                "source_file": core_source_label,
                "source_sha256_before": core_sha_before,
                "source_sha256_after": core_sha_after,
                "source_unchanged_during_both_runs": core_sha_before == core_sha_after,
                "same_runtime_contract_class_object": shared_class_object,
                "same_resolved_source_file": shared_resolved_source,
                "same_core_method_objects": shared_method_objects,
                "method_source_sha256": method_sha_after,
                "method_source_sha256_before": method_sha_before,
                "method_source_sha256_after": method_sha_after,
                "method_sources_unchanged_during_both_runs": (
                    method_sha_before == method_sha_after
                ),
                "oracle_isolation": {
                    "runtime_event_contains_expected_outcome": runtime_event_has_oracle,
                    "runtime_contract_source_reads_expected_outcome": (
                        runtime_contract_reads_oracle
                    ),
                },
                "domain_executions": {
                    "devops": {
                        "adapter_class": (
                            f"{type(devops_adapter).__module__}."
                            f"{type(devops_adapter).__qualname__}"
                        ),
                        "adapter_source_file": str(
                            devops_adapter_source.relative_to(PROJECT_ROOT)
                        ),
                        "adapter_source_sha256": _sha256_file(devops_adapter_source),
                        "runtime_contract_class": runtime_contract_name,
                        "core_source_sha256": core_sha_before,
                        "observed_core_calls": devops_summary["observed_core_calls"],
                    },
                    "uav_emergency": {
                        "adapter_class": (
                            f"{type(uav_adapter).__module__}."
                            f"{type(uav_adapter).__qualname__}"
                        ),
                        "adapter_source_file": str(
                            uav_adapter_source.relative_to(PROJECT_ROOT)
                        ),
                        "adapter_source_sha256": _sha256_file(uav_adapter_source),
                        "runtime_contract_class": runtime_contract_name,
                        "core_source_sha256": core_sha_before,
                        "observed_core_calls": uav_summary["observed_core_calls"],
                    },
                },
            },
        },
        "evidence_boundary": (
            "Deterministic in-memory DevOps and UAV workloads exercising one core contract; "
            "they are not evidence of production deployment, distributed-system throughput, "
            "or unrestricted coverage of either application domain."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEVOPS_SEED)
    parser.add_argument("--instructions", type=int, default=60)
    parser.add_argument("--events", type=int, default=60)
    parser.add_argument("--uav-seed", type=int, default=DEFAULT_UAV_CONTRACT_SEED)
    parser.add_argument("--uav-instructions", type=int, default=60)
    parser.add_argument("--uav-events", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = run_experiment(
        seed=args.seed,
        instruction_count=args.instructions,
        event_count=args.events,
        uav_seed=args.uav_seed,
        uav_instruction_count=args.uav_instructions,
        uav_event_count=args.uav_events,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import pytest

from scripts.run_devops_portability import run_experiment
import skyrescue.core_contract as core_contract_module
from skyrescue.core_contract import (
    CORE_CONTRACT_COMPONENTS,
    ContractState,
    RuntimeContract,
    RuntimeEvent,
)
from skyrescue.devops_adapter import (
    DEFAULT_DEVOPS_SEED,
    DEVOPS_ADAPTER_COMPONENTS,
    DEVOPS_ADAPTER_REPLACEMENTS,
    DevOpsAdapter,
    build_devops_events,
    build_devops_instruction_cases,
)
from skyrescue.uav_contract_adapter import (
    UAV_ADAPTER_COMPONENTS,
    UAV_ADAPTER_REPLACEMENTS,
    UAV_SOURCE_RUNTIME_EVENT_TYPES,
    UAVEmergencyAdapter,
    build_uav_events,
    build_uav_instruction_cases,
)


@pytest.fixture(scope="module")
def summary() -> dict[str, object]:
    return run_experiment()


def _first_admitted_workflow() -> tuple[RuntimeContract, object]:
    contract = RuntimeContract(DevOpsAdapter())
    case = next(
        item
        for item in build_devops_instruction_cases()
        if bool(item["expected_executable"])
    )
    result = contract.admit(str(case["case_id"]), str(case["instruction"]))
    assert result.workflow is not None
    return contract, result.workflow


def test_default_run_uses_computed_60_by_60_workload(summary: dict[str, object]) -> None:
    configuration = summary["configuration"]
    admission = summary["admission"]
    runtime = summary["runtime"]

    assert configuration["seed"] == DEFAULT_DEVOPS_SEED
    assert configuration["instruction_requests"] == 60
    assert configuration["event_requests"] == 60
    assert configuration["llm_calls"] == 0
    assert len(configuration["input_sha256"]) == 64

    assert admission["expected_executable"] == admission["admitted"] == 48
    assert admission["expected_structured_failures"] == 12
    assert admission["observed_structured_failures"] == 12
    assert admission["compile_success_rate"] == 0.8
    assert admission["admission_accuracy"] == 1.0
    assert admission["structured_failure_accuracy"] == 1.0

    assert runtime["event_requests"] == 60
    assert runtime["unique_event_ids"] == 55
    assert runtime["recoverable_events"] == runtime["repaired_events"] == 45
    assert runtime["escalation_events"] == runtime["correct_escalations"] == 10
    assert runtime["duplicate_event_requests"] == runtime["duplicates_ignored"] == 5
    assert runtime["repair_success_rate"] == 1.0
    assert runtime["escalation_accuracy"] == 1.0
    assert runtime["duplicate_event_rejection_rate"] == 1.0
    assert runtime["boundary_handling_accuracy"] == 1.0
    assert 0.0 < runtime["workflow_change_ratio"] < 1.0
    assert runtime["commitment_preservation_rate"] == 1.0


def test_receipts_and_effect_counts_come_from_executed_operations(
    summary: dict[str, object],
) -> None:
    runtime = summary["runtime"]
    expected_operations = runtime["initial_commits"] + runtime["repaired_events"]

    assert expected_operations == 93
    assert runtime["external_invocations"] == expected_operations
    assert runtime["external_effects"] == expected_operations
    assert runtime["stored_execution_receipts"] == expected_operations
    assert runtime["duplicate_external_invocations"] == 0
    assert runtime["duplicate_external_effects"] == 0
    assert runtime["receipt_reconciliations"] == 5


def test_uav_source_domain_runs_a_second_60_by_60_workload(
    summary: dict[str, object],
) -> None:
    source_domain = summary["uav_source_domain"]
    configuration = source_domain["configuration"]
    admission = source_domain["admission"]
    runtime = source_domain["runtime"]

    assert configuration["instruction_requests"] == 60
    assert configuration["event_requests"] == 60
    assert configuration["llm_calls"] == 0
    assert admission["expected_executable"] == admission["admitted"] == 48
    assert admission["expected_structured_failures"] == 12
    assert admission["observed_structured_failures"] == 12
    assert admission["admission_accuracy"] == 1.0
    assert admission["structured_failure_accuracy"] == 1.0

    assert runtime["recoverable_events"] == runtime["repaired_events"] == 45
    assert runtime["escalation_events"] == runtime["correct_escalations"] == 9
    assert runtime["duplicate_event_requests"] == runtime["duplicates_ignored"] == 6
    assert runtime["unique_event_ids"] == 54
    assert runtime["initial_commits"] == 48
    assert runtime["external_invocations"] == runtime["external_effects"] == 93
    assert runtime["stored_execution_receipts"] == 93
    assert runtime["receipt_reconciliations"] == 6
    assert runtime["duplicate_external_invocations"] == 0
    assert runtime["duplicate_external_effects"] == 0
    assert runtime["boundary_handling_accuracy"] == 1.0
    assert runtime["commitment_preservation_rate"] == 1.0


def test_fixed_seed_reproduces_cases_and_events() -> None:
    first_cases = build_devops_instruction_cases(seed=DEFAULT_DEVOPS_SEED)
    second_cases = build_devops_instruction_cases(seed=DEFAULT_DEVOPS_SEED)
    assert first_cases == second_cases

    first_contract = RuntimeContract(DevOpsAdapter())
    second_contract = RuntimeContract(DevOpsAdapter())
    first_workflows = [
        result.workflow
        for case in first_cases
        if (
            result := first_contract.admit(str(case["case_id"]), str(case["instruction"]))
        ).workflow
        is not None
    ]
    second_workflows = [
        result.workflow
        for case in second_cases
        if (
            result := second_contract.admit(str(case["case_id"]), str(case["instruction"]))
        ).workflow
        is not None
    ]
    assert build_devops_events(first_workflows) == build_devops_events(second_workflows)


def test_receipt_loss_is_reconciled_without_reinvocation() -> None:
    contract, workflow = _first_admitted_workflow()
    commit = contract.commit_external_effect(
        workflow,
        idempotency_key="receipt-loss-test",
        causal_parent="test-event",
        payload={"operation": "restart"},
        node_id="external_action",
        simulate_receipt_loss=True,
    )

    assert commit.state == ContractState.COMMITTED
    assert commit.reconciled is True
    assert commit.invoke_count == 1
    assert commit.effect_count == 1
    assert commit.receipt_count == 1
    assert contract.receiver.duplicate_invocations == 0
    assert contract.receiver.duplicate_effects == 0


def test_impact_closure_preserves_committed_nodes_outside_repair() -> None:
    contract, workflow = _first_admitted_workflow()
    contract.commit_external_effect(
        workflow,
        idempotency_key="initial-effect-test",
        causal_parent="test-admission",
        payload={"operation": "initial"},
        node_id="external_action",
    )
    protected_ids = ("intent", "policy", "resource_lock")
    protected_before = {
        node_id: copy.deepcopy(workflow.nodes[node_id]) for node_id in protected_ids
    }
    event = RuntimeEvent(
        event_id="impact-closure-test",
        workflow_id=workflow.workflow_id,
        event_type="service_failure",
        directly_affected=frozenset({"external_action"}),
        metadata={"effect_node": "external_action"},
    )

    result = contract.process_event(workflow, event)

    assert result.status == "repaired"
    assert set(result.impact_closure) == {"external_action", "verification", "audit"}
    assert result.commitment_preservation == 1.0
    for node_id, snapshot in protected_before.items():
        assert workflow.nodes[node_id] == snapshot


def test_portability_manifest_separates_core_and_adapter(
    summary: dict[str, object],
) -> None:
    portability = summary["portability_contract"]

    assert portability["reused_core_components"] == list(CORE_CONTRACT_COMPONENTS)
    assert portability["reused_core_component_count"] == len(CORE_CONTRACT_COMPONENTS)
    assert portability["replaced_adapter_components"] == DEVOPS_ADAPTER_REPLACEMENTS
    assert portability["replaced_adapter_component_count"] == len(DEVOPS_ADAPTER_COMPONENTS)
    assert set(DEVOPS_ADAPTER_COMPONENTS).isdisjoint(CORE_CONTRACT_COMPONENTS)


def test_two_adapters_execute_the_identical_core_code_object(
    summary: dict[str, object],
) -> None:
    devops_contract = RuntimeContract(DevOpsAdapter())
    uav_contract = RuntimeContract(UAVEmergencyAdapter())
    assert type(devops_contract) is type(uav_contract) is RuntimeContract
    for method_name in (
        "admit",
        "impact_closure",
        "commit_external_effect",
        "process_event",
    ):
        assert (
            getattr(type(devops_contract), method_name)
            is getattr(type(uav_contract), method_name)
            is getattr(RuntimeContract, method_name)
        )

    identity = summary["portability_contract"]["core_identity"]
    independently_computed_sha = hashlib.sha256(
        Path(core_contract_module.__file__).read_bytes()
    ).hexdigest()
    assert identity["source_sha256_before"] == independently_computed_sha
    assert identity["source_sha256_after"] == independently_computed_sha
    assert identity["source_unchanged_during_both_runs"] is True
    assert identity["same_runtime_contract_class_object"] is True
    assert identity["same_resolved_source_file"] is True
    assert identity["same_core_method_objects"] is True
    assert set(identity["method_source_sha256"]) == {
        "admit",
        "impact_closure",
        "commit_external_effect",
        "process_event",
    }
    assert identity["method_source_sha256_before"] == identity["method_source_sha256"]
    assert identity["method_source_sha256_after"] == identity["method_source_sha256"]
    assert identity["method_sources_unchanged_during_both_runs"] is True
    assert identity["oracle_isolation"] == {
        "runtime_event_contains_expected_outcome": False,
        "runtime_contract_source_reads_expected_outcome": False,
    }
    assert {
        execution["core_source_sha256"]
        for execution in identity["domain_executions"].values()
    } == {independently_computed_sha}
    assert identity["domain_executions"]["devops"]["observed_core_calls"] == {
        "admit": 60,
        "impact_closure": 55,
        "commit_external_effect": 93,
        "process_event": 60,
    }
    assert identity["domain_executions"]["uav_emergency"]["observed_core_calls"] == {
        "admit": 60,
        "impact_closure": 54,
        "commit_external_effect": 93,
        "process_event": 60,
    }


def test_adapter_replacement_manifests_are_domain_specific(
    summary: dict[str, object],
) -> None:
    replacements = summary["portability_contract"]["adapter_replacements"]
    assert replacements["devops"] == DEVOPS_ADAPTER_REPLACEMENTS
    assert replacements["uav_emergency"] == UAV_ADAPTER_REPLACEMENTS
    assert set(DEVOPS_ADAPTER_COMPONENTS) == set(UAV_ADAPTER_COMPONENTS)
    assert replacements["devops"] != replacements["uav_emergency"]


def test_uav_workload_labels_source_taxonomy_and_contract_extensions(
    summary: dict[str, object],
) -> None:
    provenance = summary["uav_source_domain"]["workload_provenance"]
    assert provenance["kind"] == "synthetic_template_generated"
    assert provenance["source_runtime_event_types"] == list(UAV_SOURCE_RUNTIME_EVENT_TYPES)
    assert provenance["contract_only_event_extensions"] == [
        "receipt_missing",
        "duplicate_event",
    ]


def test_uav_source_events_retain_frozen_impact_closure_sizes() -> None:
    contract = RuntimeContract(UAVEmergencyAdapter())
    workflows = []
    for case in build_uav_instruction_cases():
        result = contract.admit(str(case["case_id"]), str(case["instruction"]))
        if result.workflow is not None:
            workflows.append(result.workflow)
    events = build_uav_events(workflows)
    expected_sizes = {
        "new_task": 4,
        "uav_fault": 4,
        "communication_loss": 3,
        "danger_zone": 4,
        "takeoff_site_unavailable": 4,
        "priority_preemption": 4,
        "node_restart": 2,
    }
    for event_type, expected_size in expected_sizes.items():
        event = next(
            item
            for item in events
            if item.event_type == event_type
            and not item.metadata.get("approval_timed_out")
            and not item.metadata.get("concurrent_uav_fault")
        )
        workflow = next(item for item in workflows if item.workflow_id == event.workflow_id)
        assert len(contract.impact_closure(workflow, event.directly_affected)) == expected_size


def test_runtime_decision_is_structurally_isolated_from_oracle() -> None:
    assert "expected_outcome" not in RuntimeEvent.__dataclass_fields__
    assert "expected_outcome" not in inspect.getsource(RuntimeContract.process_event)
    contract, workflow = _first_admitted_workflow()
    contract.commit_external_effect(
        workflow,
        idempotency_key="oracle-isolation-initial",
        causal_parent="oracle-isolation",
        payload={"operation": "initial"},
        node_id="external_action",
    )
    repair_event = RuntimeEvent(
        event_id="oracle-isolated-repair",
        workflow_id=workflow.workflow_id,
        event_type="service_failure",
        directly_affected=frozenset({"external_action"}),
        metadata={"effect_node": "external_action"},
    )
    escalation_event = RuntimeEvent(
        event_id="oracle-isolated-escalation",
        workflow_id=workflow.workflow_id,
        event_type="permission_change",
        directly_affected=frozenset({"policy"}),
        metadata={"effect_node": "external_action"},
    )

    assert contract.process_event(workflow, repair_event).status == "repaired"
    escalation = contract.process_event(workflow, escalation_event)
    assert escalation.status == "escalated"
    assert escalation.failure is not None
    assert escalation.failure.code == "PermissionDenied"


def test_negative_cases_cover_typed_admission_failures() -> None:
    contract = RuntimeContract(DevOpsAdapter())
    failure_codes = {
        result.failure.code
        for case in build_devops_instruction_cases()
        if (
            result := contract.admit(str(case["case_id"]), str(case["instruction"]))
        ).failure
        is not None
    }
    assert failure_codes == {
        "HumanApprovalRequired",
        "MissingField",
        "PermissionDenied",
        "ResourceUnavailable",
        "TemporalConflict",
        "UngroundedEntity",
        "UnknownSkill",
    }

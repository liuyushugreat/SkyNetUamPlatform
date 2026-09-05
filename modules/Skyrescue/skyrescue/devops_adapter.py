"""Deterministic DevOps adapter for the domain-independent runtime contract."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from .core_contract import (
    Candidate,
    ContractState,
    RuntimeEvent,
    StructuredFailure,
    Workflow,
    WorkflowNode,
)


DEFAULT_DEVOPS_SEED = 20260905

DEVOPS_ADAPTER_REPLACEMENTS = {
    "instruction_parser": "strict DevOps operation-template parser",
    "entity_ontology": "service, compute-node, and deployment-slot catalogs",
    "skill_registry": "DevOps operation-to-role registry",
    "resource_binder": "service/node/slot workflow bindings",
    "domain_constraint_checker": "role, approval, resource, and slot checks",
    "runtime_event_mapper": "DevOps incident-to-impact-node mapping",
}
DEVOPS_ADAPTER_COMPONENTS = tuple(DEVOPS_ADAPTER_REPLACEMENTS)

SERVICES = (
    "auth-api",
    "billing-api",
    "catalog-api",
    "checkout-api",
    "event-worker",
    "inventory-api",
    "notification-api",
    "orders-api",
    "search-api",
    "user-api",
)
COMPUTE_NODES = ("node-a", "node-b", "node-c", "node-d", "node-e", "node-f")
DEPLOYMENT_SLOTS = tuple(f"slot-{index:02d}" for index in range(12))
SKILL_REGISTRY = {
    "restart": frozenset({"operator", "sre"}),
    "deploy": frozenset({"deployer", "sre"}),
    "scale": frozenset({"operator", "sre"}),
    "rollback": frozenset({"deployer", "sre"}),
    "migrate": frozenset({"sre"}),
}
HIGH_RISK_ACTIONS = frozenset({"deploy", "rollback", "migrate"})

DEVOPS_EVENT_TYPES = (
    "service_failure",
    "node_unavailable",
    "deployment_conflict",
    "permission_change",
    "restart",
    "rollback",
    "resource_exhaustion",
    "higher_priority_incident",
    "approval_timeout",
    "receipt_missing",
    "duplicate_event",
    "concurrent_failure",
)

_INSTRUCTION = re.compile(
    r"^(?P<action>[a-z-]+) service (?P<target>[a-z0-9-]+)"
    r"(?: on (?P<resource>[a-z0-9-]+))? in (?P<slot>slot-[0-9]{2})"
    r" as (?P<permission>[a-z-]+) approval (?P<approval>granted|absent)\.$"
)


def _next_value(current: str, values: Sequence[str]) -> str:
    try:
        index = values.index(current)
    except ValueError:
        return values[0]
    return values[(index + 1) % len(values)]


class DevOpsAdapter:
    """Replace only domain parsing, ontology, binding, and repair policy."""

    domain_name = "devops_incident_response"
    replaced_components = DEVOPS_ADAPTER_COMPONENTS

    def parse_instruction(self, case_id: str, instruction: str) -> Candidate | StructuredFailure:
        match = _INSTRUCTION.fullmatch(instruction.strip())
        if match is None or match.group("resource") is None:
            return StructuredFailure("MissingField", "A service, node, slot, role, and approval are required.")
        values = match.groupdict()
        action = values["action"]
        return Candidate(
            candidate_id=case_id,
            action=action,
            target=values["target"],
            resource=values["resource"],
            slot=values["slot"],
            permission=values["permission"],
            requires_approval=action in HIGH_RISK_ACTIONS,
            approval_granted=values["approval"] == "granted",
            parameters={"source": "deterministic_devops_instruction"},
        )

    def adjudicate(self, candidate: Candidate) -> StructuredFailure | None:
        if candidate.action not in SKILL_REGISTRY:
            return StructuredFailure("UnknownSkill", f"Unregistered operation: {candidate.action}")
        if candidate.target not in SERVICES:
            return StructuredFailure("UngroundedEntity", f"Unknown service: {candidate.target}")
        if candidate.resource not in COMPUTE_NODES:
            return StructuredFailure("ResourceUnavailable", f"Unavailable node: {candidate.resource}")
        if candidate.slot not in DEPLOYMENT_SLOTS:
            return StructuredFailure("TemporalConflict", f"Unknown deployment slot: {candidate.slot}")
        if candidate.permission not in SKILL_REGISTRY[candidate.action]:
            return StructuredFailure(
                "PermissionDenied",
                f"Role {candidate.permission} cannot execute {candidate.action}",
            )
        if candidate.requires_approval and not candidate.approval_granted:
            return StructuredFailure(
                "HumanApprovalRequired",
                f"Operation {candidate.action} requires production approval",
            )
        return None

    def build_workflow(self, candidate: Candidate) -> Workflow:
        nodes = {
            "intent": WorkflowNode(
                "intent",
                "typed_task",
                ContractState.COMMITTED,
                {"target": candidate.target, "action": candidate.action},
            ),
            "policy": WorkflowNode(
                "policy",
                "permission_gate",
                ContractState.COMMITTED,
                {
                    "permission": candidate.permission,
                    "approval": candidate.approval_granted,
                },
            ),
            "resource_lock": WorkflowNode(
                "resource_lock",
                "deployment_lock",
                ContractState.COMMITTED,
                {
                    "target": candidate.target,
                    "resource": candidate.resource,
                    "slot": candidate.slot,
                },
            ),
            "external_action": WorkflowNode(
                "external_action",
                "external_action",
                ContractState.PRECHECKED,
                {
                    "target": candidate.target,
                    "action": candidate.action,
                    "resource": candidate.resource,
                    "slot": candidate.slot,
                },
                committed=False,
            ),
            "verification": WorkflowNode(
                "verification",
                "health_check",
                ContractState.PROPOSED,
                {"target": candidate.target, "check": "healthy"},
                committed=False,
            ),
            "audit": WorkflowNode(
                "audit",
                "audit_record",
                ContractState.PROPOSED,
                {"target": candidate.target},
                committed=False,
            ),
        }
        return Workflow(
            workflow_id=f"devops-{candidate.candidate_id}",
            nodes=nodes,
            causal_edges={
                "intent": {"policy"},
                "policy": {"resource_lock"},
                "resource_lock": {"external_action"},
                "external_action": {"verification"},
                "verification": {"audit"},
                "audit": set(),
            },
        )

    def plan_repair(
        self,
        workflow: Workflow,
        event: RuntimeEvent,
        impact_closure: frozenset[str],
    ) -> dict[str, dict[str, object]] | StructuredFailure:
        del impact_closure
        event_type = event.event_type
        resource = str(workflow.nodes["resource_lock"].binding["resource"])
        slot = str(workflow.nodes["resource_lock"].binding["slot"])
        next_resource = str(event.metadata.get("replacement_resource", _next_value(resource, COMPUTE_NODES)))
        next_slot = str(event.metadata.get("replacement_slot", _next_value(slot, DEPLOYMENT_SLOTS)))
        audit_patch = {"last_event": event.event_id, "event_type": event_type}

        if event_type == "permission_change":
            return StructuredFailure(
                "PermissionDenied",
                "The changed production permission requires external adjudication.",
            )
        if event_type == "approval_timeout":
            return StructuredFailure(
                "HumanApprovalRequired",
                "The production approval expired before the repair could commit.",
            )
        if event_type == "service_failure":
            return {
                "external_action": {
                    "recovery_operation": "restart",
                    "last_event": event.event_id,
                },
                "verification": {"check": "restart_healthy"},
                "audit": audit_patch,
            }
        if event_type in {"node_unavailable", "resource_exhaustion"}:
            return {
                "resource_lock": {"resource": next_resource},
                "external_action": {"resource": next_resource, "last_event": event.event_id},
                "verification": {"check": "replacement_healthy"},
                "audit": audit_patch,
            }
        if event_type == "deployment_conflict":
            return {
                "resource_lock": {"slot": next_slot},
                "external_action": {"slot": next_slot, "last_event": event.event_id},
                "audit": audit_patch,
            }
        if event_type == "restart":
            return {
                "external_action": {
                    "recovery_operation": "restart",
                    "last_event": event.event_id,
                },
                "verification": {"check": "restart_healthy"},
                "audit": audit_patch,
            }
        if event_type == "rollback":
            return {
                "external_action": {
                    "recovery_operation": "rollback",
                    "last_event": event.event_id,
                },
                "verification": {"check": "rollback_healthy"},
                "audit": audit_patch,
            }
        if event_type == "higher_priority_incident":
            return {
                "resource_lock": {"slot": next_slot, "priority": "critical"},
                "external_action": {"slot": next_slot, "priority": "critical", "last_event": event.event_id},
                "audit": audit_patch,
            }
        if event_type == "receipt_missing":
            return {
                "external_action": {"last_event": event.event_id},
                "verification": {"check": "receipt_reconciled"},
                "audit": audit_patch,
            }
        if event_type == "concurrent_failure":
            return {
                "resource_lock": {"resource": next_resource, "slot": next_slot},
                "external_action": {
                    "resource": next_resource,
                    "slot": next_slot,
                    "last_event": event.event_id,
                },
                "verification": {"check": "concurrent_recovery_healthy"},
                "audit": audit_patch,
            }
        return StructuredFailure("NoFeasiblePlan", f"No local repair for event: {event_type}")

    def check_invariants(self, workflow: Workflow) -> StructuredFailure | None:
        expected_nodes = {
            "intent",
            "policy",
            "resource_lock",
            "external_action",
            "verification",
            "audit",
        }
        if set(workflow.nodes) != expected_nodes:
            return StructuredFailure("InvalidWorkflow", "The DevOps workflow node set is incomplete.")
        lock = workflow.nodes["resource_lock"].binding
        action = workflow.nodes["external_action"].binding
        intent = workflow.nodes["intent"].binding
        policy = workflow.nodes["policy"].binding
        if lock.get("target") not in SERVICES or action.get("target") != lock.get("target"):
            return StructuredFailure("UngroundedEntity", "Action and lock must reference one known service.")
        if intent.get("target") != lock.get("target") or intent.get("action") != action.get("action"):
            return StructuredFailure("InvalidWorkflow", "Committed intent and external action must agree.")
        if lock.get("resource") not in COMPUTE_NODES or action.get("resource") != lock.get("resource"):
            return StructuredFailure("ResourceUnavailable", "Action and lock require one available node.")
        if lock.get("slot") not in DEPLOYMENT_SLOTS or action.get("slot") != lock.get("slot"):
            return StructuredFailure("TemporalConflict", "Action and lock require one valid deployment slot.")
        if action.get("action") not in SKILL_REGISTRY:
            return StructuredFailure("UnknownSkill", "The repaired action is not registered.")
        registered_action = str(action["action"])
        if policy.get("permission") not in SKILL_REGISTRY[registered_action]:
            return StructuredFailure("PermissionDenied", "The committed role cannot execute the action.")
        if registered_action in HIGH_RISK_ACTIONS and not policy.get("approval"):
            return StructuredFailure("HumanApprovalRequired", "The committed approval no longer suffices.")
        recovery_operation = action.get("recovery_operation")
        if recovery_operation is not None and recovery_operation not in {"restart", "rollback"}:
            return StructuredFailure("UnknownSkill", "The recovery operation is not registered.")
        referenced = set().union(*workflow.causal_edges.values()) if workflow.causal_edges else set()
        if referenced - set(workflow.nodes):
            return StructuredFailure("InvalidWorkflow", "A causal edge references an unknown node.")
        return None


def _instruction(action: str, service: str, node: str, slot: str, role: str, approval: str) -> str:
    return f"{action} service {service} on {node} in {slot} as {role} approval {approval}."


def build_devops_instruction_cases(
    count: int = 60,
    *,
    seed: int = DEFAULT_DEVOPS_SEED,
) -> list[dict[str, object]]:
    """Create 80% valid and 20% negative deterministic admission cases."""

    if not 50 <= count <= 100:
        raise ValueError("DevOps portability uses between 50 and 100 instructions")
    rng = random.Random(seed)
    invalid_count = max(1, count // 5)
    valid_count = count - invalid_count
    actions = tuple(SKILL_REGISTRY)
    cases: list[dict[str, object]] = []

    for index in range(valid_count):
        action = actions[index % len(actions)]
        service = SERVICES[(index * 3 + rng.randrange(len(SERVICES))) % len(SERVICES)]
        node = COMPUTE_NODES[(index + rng.randrange(len(COMPUTE_NODES))) % len(COMPUTE_NODES)]
        slot = DEPLOYMENT_SLOTS[(index * 2 + rng.randrange(len(DEPLOYMENT_SLOTS))) % len(DEPLOYMENT_SLOTS)]
        role = sorted(SKILL_REGISTRY[action])[index % len(SKILL_REGISTRY[action])]
        approval = "granted" if action in HIGH_RISK_ACTIONS else "absent"
        cases.append(
            {
                "case_id": f"DVI{index + 1:03d}",
                "instruction": _instruction(action, service, node, slot, role, approval),
                "expected_executable": True,
                "expected_failure": None,
            }
        )

    invalid_builders = (
        lambda i: (_instruction("destroy", SERVICES[i % len(SERVICES)], "node-a", "slot-00", "sre", "granted"), "UnknownSkill"),
        lambda i: (_instruction("restart", "ghost-service", "node-a", "slot-00", "sre", "absent"), "UngroundedEntity"),
        lambda i: ("restart service auth-api in slot-00 as sre approval absent.", "MissingField"),
        lambda i: (_instruction("restart", SERVICES[i % len(SERVICES)], "node-a", "slot-00", "viewer", "absent"), "PermissionDenied"),
        lambda i: (_instruction("deploy", SERVICES[i % len(SERVICES)], "node-a", "slot-00", "deployer", "absent"), "HumanApprovalRequired"),
        lambda i: (_instruction("restart", SERVICES[i % len(SERVICES)], "node-offline", "slot-00", "sre", "absent"), "ResourceUnavailable"),
        lambda i: (_instruction("restart", SERVICES[i % len(SERVICES)], "node-a", "slot-99", "sre", "absent"), "TemporalConflict"),
    )
    for offset in range(invalid_count):
        instruction, failure = invalid_builders[offset % len(invalid_builders)](offset)
        cases.append(
            {
                "case_id": f"DVI{valid_count + offset + 1:03d}",
                "instruction": instruction,
                "expected_executable": False,
                "expected_failure": failure,
            }
        )
    rng.shuffle(cases)
    return cases


def build_devops_events(
    workflows: Sequence[Workflow],
    count: int = 60,
    *,
    seed: int = DEFAULT_DEVOPS_SEED,
) -> list[RuntimeEvent]:
    """Create deterministic runtime inputs without evaluation labels."""

    if not workflows:
        raise ValueError("At least one admitted workflow is required")
    if not 50 <= count <= 100:
        raise ValueError("DevOps portability uses between 50 and 100 event requests")
    rng = random.Random(seed ^ 0x5A17)
    events: list[RuntimeEvent] = []
    duplicate_source: RuntimeEvent | None = None
    affected_by_type = {
        "service_failure": frozenset({"external_action"}),
        "node_unavailable": frozenset({"resource_lock"}),
        "deployment_conflict": frozenset({"resource_lock"}),
        "permission_change": frozenset({"policy"}),
        "restart": frozenset({"external_action"}),
        "rollback": frozenset({"external_action"}),
        "resource_exhaustion": frozenset({"resource_lock"}),
        "higher_priority_incident": frozenset({"resource_lock"}),
        "approval_timeout": frozenset({"policy"}),
        "receipt_missing": frozenset({"external_action"}),
        "concurrent_failure": frozenset({"resource_lock"}),
    }

    for index in range(count):
        event_type = DEVOPS_EVENT_TYPES[index % len(DEVOPS_EVENT_TYPES)]
        if event_type == "duplicate_event":
            if duplicate_source is None:
                raise RuntimeError("Duplicate event must follow its source event")
            events.append(
                RuntimeEvent(
                    event_id=duplicate_source.event_id,
                    workflow_id=duplicate_source.workflow_id,
                    event_type="duplicate_event",
                    directly_affected=duplicate_source.directly_affected,
                    metadata={"replay_of": duplicate_source.event_id},
                )
            )
            continue

        workflow = workflows[(index * 7 + rng.randrange(len(workflows))) % len(workflows)]
        event_id = f"DVE{index + 1:03d}"
        resource = str(workflow.nodes["resource_lock"].binding["resource"])
        slot = str(workflow.nodes["resource_lock"].binding["slot"])
        event = RuntimeEvent(
            event_id=event_id,
            workflow_id=workflow.workflow_id,
            event_type=event_type,
            directly_affected=affected_by_type[event_type],
            metadata={
                "replacement_resource": _next_value(resource, COMPUTE_NODES),
                "replacement_slot": _next_value(slot, DEPLOYMENT_SLOTS),
                "effect_node": "external_action",
            },
        )
        events.append(event)
        if event_type == "service_failure":
            duplicate_source = event
    return events

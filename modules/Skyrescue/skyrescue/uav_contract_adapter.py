"""Synthetic template-generated UAV adapter for the shared runtime contract.

This adapter deliberately contains only source-domain knowledge.  Admission,
idempotent effect commitment, receipt reconciliation, impact closure, and
commitment protection remain implemented by :mod:`skyrescue.core_contract`.
It reuses frozen SkyRescue vocabularies but is not a production UAV deployment.
"""

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
from .security import ACTION_PERMISSIONS
from .workflow import RUNTIME_EVENT_TYPES, TASK_TYPES, ZONE_ALIASES


DEFAULT_UAV_CONTRACT_SEED = 20260905

UAV_ADAPTER_REPLACEMENTS = {
    "instruction_parser": "strict UAV emergency-mission template parser",
    "entity_ontology": "SkyRescue task and emergency-zone catalogs",
    "skill_registry": "mission-skill and action-permission registry",
    "resource_binder": "UAV/airspace-window workflow bindings",
    "domain_constraint_checker": "skill, authority, UAV, and airspace checks",
    "runtime_event_mapper": "UAV incident-to-impact-node mapping",
}
UAV_ADAPTER_COMPONENTS = tuple(UAV_ADAPTER_REPLACEMENTS)

UAV_MISSION_SKILLS = {
    "medical-delivery": TASK_TYPES["MedicalDelivery"]["skill"],
    "communication-relay": TASK_TYPES["CommunicationRelay"]["skill"],
    "search": TASK_TYPES["Search"]["skill"],
    "mapping": TASK_TYPES["Mapping"]["skill"],
    "evacuation-coordination": TASK_TYPES["EvacuationCoordination"]["skill"],
    "cargo-delivery": TASK_TYPES["CargoDelivery"]["skill"],
}
UAV_DISPATCH_PERMISSION = ACTION_PERMISSIONS["dispatch_uav"]
UAV_HIGH_RISK_ACTIONS = frozenset({"evacuation-coordination"})
UAV_RESOURCES = {
    "U0001": frozenset({"medical_payload", "cargo"}),
    "U0002": frozenset({"relay", "coordination"}),
    "U0003": frozenset({"camera", "mapping"}),
    "U0004": frozenset(UAV_MISSION_SKILLS.values()),
    "U0005": frozenset({"medical_payload", "camera", "cargo"}),
    "U0006": frozenset({"relay", "mapping", "coordination"}),
}
UAV_FLIGHT_WINDOWS = tuple(f"window-{index:02d}" for index in range(12))
UAV_SOURCE_RUNTIME_EVENT_TYPES = tuple(RUNTIME_EVENT_TYPES)
UAV_CONTRACT_EVENT_PROFILES = (
    *((event_type, None) for event_type in UAV_SOURCE_RUNTIME_EVENT_TYPES),
    ("receipt_missing", None),
    ("duplicate_event", None),
    ("danger_zone", "concurrent_uav_fault"),
)

_INSTRUCTION = re.compile(
    r"^dispatch (?P<action>[a-z-]+) to (?P<target>[A-Za-z0-9-]+)"
    r"(?: with (?P<resource>U[0-9]{4}))? in (?P<slot>window-[0-9]{2})"
    r" permission (?P<permission>[a-z.]+) approval (?P<approval>granted|absent)\.$"
)


def _next_value(current: str, values: Sequence[str]) -> str:
    try:
        index = values.index(current)
    except ValueError:
        return values[0]
    return values[(index + 1) % len(values)]


def _compatible_resources(action: str) -> tuple[str, ...]:
    required_skill = UAV_MISSION_SKILLS[action]
    return tuple(
        resource
        for resource, skills in UAV_RESOURCES.items()
        if required_skill in skills
    )


class UAVEmergencyAdapter:
    """Bind the shared contract to SkyRescue emergency-mission concepts."""

    domain_name = "uav_emergency_response"
    replaced_components = UAV_ADAPTER_COMPONENTS

    def parse_instruction(self, case_id: str, instruction: str) -> Candidate | StructuredFailure:
        match = _INSTRUCTION.fullmatch(instruction.strip())
        if match is None or match.group("resource") is None:
            return StructuredFailure(
                "MissingField",
                "A mission, zone, UAV, window, permission, and approval are required.",
            )
        values = match.groupdict()
        action = values["action"]
        return Candidate(
            candidate_id=case_id,
            action=action,
            target=values["target"],
            resource=values["resource"],
            slot=values["slot"],
            permission=values["permission"],
            requires_approval=action in UAV_HIGH_RISK_ACTIONS,
            approval_granted=values["approval"] == "granted",
            parameters={
                "source": "deterministic_uav_emergency_instruction",
                "source_trust": "authenticated",
            },
        )

    def adjudicate(self, candidate: Candidate) -> StructuredFailure | None:
        if candidate.action not in UAV_MISSION_SKILLS:
            return StructuredFailure("UnknownSkill", f"Unknown mission: {candidate.action}")
        if candidate.target not in ZONE_ALIASES:
            return StructuredFailure("UngroundedEntity", f"Unknown emergency zone: {candidate.target}")
        if candidate.resource not in UAV_RESOURCES:
            return StructuredFailure("ResourceUnavailable", f"Unavailable UAV: {candidate.resource}")
        required_skill = UAV_MISSION_SKILLS[candidate.action]
        if required_skill not in UAV_RESOURCES[candidate.resource]:
            return StructuredFailure(
                "ResourceUnavailable",
                f"UAV {candidate.resource} lacks {required_skill}",
            )
        if candidate.slot not in UAV_FLIGHT_WINDOWS:
            return StructuredFailure("TemporalConflict", f"Unknown flight window: {candidate.slot}")
        if candidate.permission != UAV_DISPATCH_PERMISSION:
            return StructuredFailure(
                "PermissionDenied",
                f"Permission {candidate.permission} cannot dispatch a UAV",
            )
        if candidate.requires_approval and not candidate.approval_granted:
            return StructuredFailure(
                "HumanApprovalRequired",
                f"Mission {candidate.action} requires incident-command approval",
            )
        return None

    def build_workflow(self, candidate: Candidate) -> Workflow:
        required_skill = UAV_MISSION_SKILLS[candidate.action]
        nodes = {
            "mission_intent": WorkflowNode(
                "mission_intent",
                "typed_mission",
                ContractState.COMMITTED,
                {
                    "target": candidate.target,
                    "action": candidate.action,
                    "skill": required_skill,
                    "priority": "high",
                },
            ),
            "authority_gate": WorkflowNode(
                "authority_gate",
                "permission_gate",
                ContractState.COMMITTED,
                {
                    "permission": candidate.permission,
                    "approval": candidate.approval_granted,
                    "source_trust": candidate.parameters["source_trust"],
                },
            ),
            "airspace_reservation": WorkflowNode(
                "airspace_reservation",
                "resource_and_airspace_lock",
                ContractState.COMMITTED,
                {
                    "target": candidate.target,
                    "resource": candidate.resource,
                    "slot": candidate.slot,
                    "launch_site": "primary",
                },
            ),
            "dispatch_effect": WorkflowNode(
                "dispatch_effect",
                "external_action",
                ContractState.PRECHECKED,
                {
                    "target": candidate.target,
                    "action": candidate.action,
                    "skill": required_skill,
                    "resource": candidate.resource,
                    "slot": candidate.slot,
                },
                committed=False,
            ),
            "mission_monitor": WorkflowNode(
                "mission_monitor",
                "mission_health_check",
                ContractState.PROPOSED,
                {"target": candidate.target, "check": "mission_healthy"},
                committed=False,
            ),
            "evidence_log": WorkflowNode(
                "evidence_log",
                "audit_record",
                ContractState.PROPOSED,
                {"target": candidate.target},
                committed=False,
            ),
        }
        return Workflow(
            workflow_id=f"uav-{candidate.candidate_id}",
            nodes=nodes,
            causal_edges={
                "mission_intent": {"authority_gate"},
                "authority_gate": {"airspace_reservation"},
                "airspace_reservation": {"dispatch_effect"},
                "dispatch_effect": {"mission_monitor"},
                "mission_monitor": {"evidence_log"},
                "evidence_log": set(),
            },
        )

    def plan_repair(
        self,
        workflow: Workflow,
        event: RuntimeEvent,
        impact_closure: frozenset[str],
    ) -> dict[str, dict[str, object]] | StructuredFailure:
        del impact_closure
        dispatch = workflow.nodes["dispatch_effect"].binding
        action = str(dispatch["action"])
        resource = str(dispatch["resource"])
        slot = str(dispatch["slot"])
        compatible = _compatible_resources(action)
        replacement_resource = str(
            event.metadata.get("replacement_resource", _next_value(resource, compatible))
        )
        replacement_slot = str(
            event.metadata.get("replacement_slot", _next_value(slot, UAV_FLIGHT_WINDOWS))
        )
        log_patch = {"last_event": event.event_id, "event_type": event.event_type}

        if event.event_type == "priority_preemption" and event.metadata.get(
            "approval_timed_out"
        ):
            return StructuredFailure(
                "HumanApprovalTimeout",
                "Priority repair exceeded its incident-command approval window.",
            )
        if event.event_type == "danger_zone" and event.metadata.get(
            "concurrent_uav_fault"
        ):
            return StructuredFailure(
                "ConcurrentFaultAndDangerZone",
                "Concurrent vehicle and airspace faults exceed the local repair boundary.",
            )
        if event.event_type == "new_task":
            return {
                "airspace_reservation": {"queue_mode": "priority"},
                "dispatch_effect": {"queue_mode": "priority", "last_event": event.event_id},
                "mission_monitor": {"check": "priority_mission_healthy"},
                "evidence_log": log_patch,
            }
        if event.event_type == "uav_fault":
            return {
                "airspace_reservation": {"resource": replacement_resource},
                "dispatch_effect": {
                    "resource": replacement_resource,
                    "last_event": event.event_id,
                },
                "mission_monitor": {"check": "replacement_uav_healthy"},
                "evidence_log": log_patch,
            }
        if event.event_type == "communication_loss":
            return {
                "dispatch_effect": {
                    "recovery_operation": "relay_handoff",
                    "last_event": event.event_id,
                },
                "mission_monitor": {"check": "link_restored"},
                "evidence_log": log_patch,
            }
        if event.event_type == "danger_zone":
            return {
                "airspace_reservation": {
                    "slot": replacement_slot,
                    "route_revision": "avoid_danger_zone",
                },
                "dispatch_effect": {
                    "slot": replacement_slot,
                    "route_revision": "avoid_danger_zone",
                    "last_event": event.event_id,
                },
                "mission_monitor": {"check": "replanned_route_healthy"},
                "evidence_log": log_patch,
            }
        if event.event_type == "takeoff_site_unavailable":
            return {
                "airspace_reservation": {"launch_site": "alternate"},
                "dispatch_effect": {
                    "launch_site": "alternate",
                    "last_event": event.event_id,
                },
                "evidence_log": log_patch,
            }
        if event.event_type == "priority_preemption":
            return {
                "airspace_reservation": {"queue_mode": "preempted_and_rebound"},
                "dispatch_effect": {
                    "queue_mode": "preempted_and_rebound",
                    "last_event": event.event_id,
                },
                "mission_monitor": {"check": "priority_rebind_healthy"},
                "evidence_log": log_patch,
            }
        if event.event_type == "node_restart":
            return {
                "mission_monitor": {
                    "recovery_operation": "resume_from_checkpoint",
                    "check": "checkpoint_restored",
                    "last_event": event.event_id,
                },
                "evidence_log": log_patch,
            }
        if event.event_type == "receipt_missing":
            return {
                "dispatch_effect": {"last_event": event.event_id},
                "mission_monitor": {"check": "receipt_reconciled"},
                "evidence_log": log_patch,
            }
        return StructuredFailure(
            "NoFeasiblePlan",
            f"No local UAV repair for event: {event.event_type}",
        )

    def check_invariants(self, workflow: Workflow) -> StructuredFailure | None:
        expected_nodes = {
            "mission_intent",
            "authority_gate",
            "airspace_reservation",
            "dispatch_effect",
            "mission_monitor",
            "evidence_log",
        }
        if set(workflow.nodes) != expected_nodes:
            return StructuredFailure("InvalidWorkflow", "The UAV workflow node set is incomplete.")
        intent = workflow.nodes["mission_intent"].binding
        authority = workflow.nodes["authority_gate"].binding
        reservation = workflow.nodes["airspace_reservation"].binding
        dispatch = workflow.nodes["dispatch_effect"].binding
        action = str(dispatch.get("action"))
        if action not in UAV_MISSION_SKILLS:
            return StructuredFailure("UnknownSkill", "The mission action is not registered.")
        if intent.get("action") != action or intent.get("target") != dispatch.get("target"):
            return StructuredFailure("InvalidWorkflow", "Intent and dispatch bindings must agree.")
        if dispatch.get("target") not in ZONE_ALIASES:
            return StructuredFailure("UngroundedEntity", "The dispatch target is not grounded.")
        if reservation.get("target") != dispatch.get("target"):
            return StructuredFailure("InvalidWorkflow", "Reservation and dispatch targets differ.")
        resource = str(reservation.get("resource"))
        required_skill = UAV_MISSION_SKILLS[action]
        if resource not in UAV_RESOURCES or required_skill not in UAV_RESOURCES[resource]:
            return StructuredFailure("ResourceUnavailable", "The bound UAV lacks the mission skill.")
        if dispatch.get("resource") != resource or dispatch.get("skill") != required_skill:
            return StructuredFailure("InvalidWorkflow", "Dispatch and resource bindings differ.")
        slot = reservation.get("slot")
        if slot not in UAV_FLIGHT_WINDOWS or dispatch.get("slot") != slot:
            return StructuredFailure("TemporalConflict", "The airspace window is invalid.")
        if authority.get("permission") != UAV_DISPATCH_PERMISSION:
            return StructuredFailure(
                "PermissionDenied",
                "The action permission cannot dispatch the mission.",
            )
        if authority.get("source_trust") != "authenticated":
            return StructuredFailure("PermissionDenied", "The mission source is not authenticated.")
        if action in UAV_HIGH_RISK_ACTIONS and not authority.get("approval"):
            return StructuredFailure("HumanApprovalRequired", "Incident-command approval is required.")
        referenced = set().union(*workflow.causal_edges.values()) if workflow.causal_edges else set()
        if referenced - set(workflow.nodes):
            return StructuredFailure("InvalidWorkflow", "A causal edge references an unknown node.")
        return None


def _instruction(
    action: str,
    target: str,
    resource: str,
    slot: str,
    permission: str,
    approval: str,
) -> str:
    return (
        f"dispatch {action} to {target} with {resource} in {slot} "
        f"permission {permission} approval {approval}."
    )


def build_uav_instruction_cases(
    count: int = 60,
    *,
    seed: int = DEFAULT_UAV_CONTRACT_SEED,
) -> list[dict[str, object]]:
    """Create synthetic deterministic cases from source-domain vocabularies."""

    if not 50 <= count <= 100:
        raise ValueError("UAV cross-domain evidence uses between 50 and 100 instructions")
    rng = random.Random(seed ^ 0xA117)
    invalid_count = max(1, count // 5)
    valid_count = count - invalid_count
    actions = tuple(UAV_MISSION_SKILLS)
    targets = tuple(ZONE_ALIASES)
    cases: list[dict[str, object]] = []

    for index in range(valid_count):
        action = actions[index % len(actions)]
        resources = _compatible_resources(action)
        target = targets[(index * 3 + rng.randrange(len(targets))) % len(targets)]
        resource = resources[(index + rng.randrange(len(resources))) % len(resources)]
        slot = UAV_FLIGHT_WINDOWS[(index * 2 + rng.randrange(len(UAV_FLIGHT_WINDOWS))) % len(UAV_FLIGHT_WINDOWS)]
        approval = "granted" if action in UAV_HIGH_RISK_ACTIONS else "absent"
        cases.append(
            {
                "case_id": f"UCI{index + 1:03d}",
                "instruction": _instruction(
                    action,
                    target,
                    resource,
                    slot,
                    UAV_DISPATCH_PERMISSION,
                    approval,
                ),
                "expected_executable": True,
                "expected_failure": None,
            }
        )

    invalid_builders = (
        lambda i: (_instruction("underwater-rescue", targets[i % len(targets)], "U0004", "window-00", UAV_DISPATCH_PERMISSION, "granted"), "UnknownSkill"),
        lambda i: (_instruction("search", "Zone-X-99", "U0004", "window-00", UAV_DISPATCH_PERMISSION, "absent"), "UngroundedEntity"),
        lambda i: ("dispatch search to Zone-SE-07 in window-00 permission mission.dispatch approval absent.", "MissingField"),
        lambda i: (_instruction("search", targets[i % len(targets)], "U0004", "window-00", ACTION_PERMISSIONS["reserve_airspace"], "absent"), "PermissionDenied"),
        lambda i: (_instruction("evacuation-coordination", targets[i % len(targets)], "U0004", "window-00", UAV_DISPATCH_PERMISSION, "absent"), "HumanApprovalRequired"),
        lambda i: (_instruction("search", targets[i % len(targets)], "U9999", "window-00", UAV_DISPATCH_PERMISSION, "absent"), "ResourceUnavailable"),
        lambda i: (_instruction("search", targets[i % len(targets)], "U0004", "window-99", UAV_DISPATCH_PERMISSION, "absent"), "TemporalConflict"),
        lambda i: (_instruction("mapping", targets[i % len(targets)], "U0001", "window-00", UAV_DISPATCH_PERMISSION, "absent"), "ResourceUnavailable"),
    )
    for offset in range(invalid_count):
        instruction, failure = invalid_builders[offset % len(invalid_builders)](offset)
        cases.append(
            {
                "case_id": f"UCI{valid_count + offset + 1:03d}",
                "instruction": instruction,
                "expected_executable": False,
                "expected_failure": failure,
            }
        )
    rng.shuffle(cases)
    return cases


def build_uav_events(
    workflows: Sequence[Workflow],
    count: int = 60,
    *,
    seed: int = DEFAULT_UAV_CONTRACT_SEED,
) -> list[RuntimeEvent]:
    """Create synthetic deterministic inputs without evaluation labels."""

    if not workflows:
        raise ValueError("At least one admitted UAV workflow is required")
    if not 50 <= count <= 100:
        raise ValueError("UAV cross-domain evidence uses between 50 and 100 event requests")
    rng = random.Random(seed ^ 0xE717)
    events: list[RuntimeEvent] = []
    duplicate_source: RuntimeEvent | None = None
    affected_by_type = {
        "new_task": frozenset({"airspace_reservation"}),
        "uav_fault": frozenset({"airspace_reservation"}),
        "communication_loss": frozenset({"dispatch_effect"}),
        "danger_zone": frozenset({"airspace_reservation"}),
        "takeoff_site_unavailable": frozenset({"airspace_reservation"}),
        "priority_preemption": frozenset({"airspace_reservation"}),
        "node_restart": frozenset({"mission_monitor"}),
        "receipt_missing": frozenset({"dispatch_effect"}),
    }

    for index in range(count):
        profile_index = index % len(UAV_CONTRACT_EVENT_PROFILES)
        cycle_index = index // len(UAV_CONTRACT_EVENT_PROFILES)
        event_type, injected_condition = UAV_CONTRACT_EVENT_PROFILES[profile_index]
        if event_type == "duplicate_event":
            if duplicate_source is None:
                raise RuntimeError("Duplicate UAV event must follow its source event")
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

        workflow = workflows[(index * 5 + rng.randrange(len(workflows))) % len(workflows)]
        event_id = f"UCE{index + 1:03d}"
        approval_timed_out = event_type == "priority_preemption" and cycle_index % 2 == 1
        concurrent_uav_fault = injected_condition == "concurrent_uav_fault"
        dispatch = workflow.nodes["dispatch_effect"].binding
        resource = str(dispatch["resource"])
        action = str(dispatch["action"])
        slot = str(dispatch["slot"])
        event = RuntimeEvent(
            event_id=event_id,
            workflow_id=workflow.workflow_id,
            event_type=event_type,
            directly_affected=affected_by_type[event_type],
            metadata={
                "approval_timed_out": approval_timed_out,
                "concurrent_uav_fault": concurrent_uav_fault,
                "replacement_resource": _next_value(resource, _compatible_resources(action)),
                "replacement_slot": _next_value(slot, UAV_FLIGHT_WINDOWS),
                "effect_node": (
                    "mission_monitor" if event_type == "node_restart" else "dispatch_effect"
                ),
            },
        )
        events.append(event)
        if event_type == "uav_fault":
            duplicate_source = event
    return events

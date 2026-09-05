"""Deterministic runtime paths used by the SkyRescue latency benchmark.

The functions in this module deliberately accept one frozen intent case and
one observable runtime event. Every call rebuilds its mutable state so a
previous measurement cannot affect the next one.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .core_contract import IdempotentReceiver
from .workflow import (
    TASK_TYPES,
    ZONE_ALIASES,
    _extract_zone,
    _parse_tasks,
    _workflow_nodes,
    adjudicate_runtime_event,
    compile_case,
)


def candidate_from_frozen_case(case: dict[str, Any]) -> dict[str, Any]:
    """Materialize the already-generated candidate outside the timed region."""

    tasks = _parse_tasks(case["instruction"])
    zone, unknown_zone = _extract_zone(case["instruction"])
    return {
        "candidate_id": case["case_id"],
        "tasks": tasks,
        "zone": zone,
        "unknown_zone": unknown_zone,
        "unknown_skill": "水下机器人" in case["instruction"] or "爆破机器人" in case["instruction"],
        "requires_human_approval": bool(case.get("requires_human_approval")),
        "approval_granted": bool(case.get("approval_granted")),
    }


def compile_typed_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Validate types/grounding/skills and return a workflow or failure."""

    required = {
        "candidate_id",
        "tasks",
        "zone",
        "unknown_zone",
        "unknown_skill",
        "requires_human_approval",
        "approval_granted",
    }
    if set(candidate) != required or not isinstance(candidate["tasks"], list):
        return {"kind": "StructuredFailure", "failure": "InvalidType"}
    tasks = candidate["tasks"]
    if not tasks or candidate["zone"] is None:
        return {"kind": "StructuredFailure", "failure": "MissingField"}
    if candidate["unknown_zone"] or candidate["zone"] not in ZONE_ALIASES:
        return {"kind": "StructuredFailure", "failure": "UngroundedEntity"}
    registered_skills = {spec["skill"] for spec in TASK_TYPES.values()}
    if candidate["unknown_skill"] or any(task.get("skill") not in registered_skills for task in tasks):
        return {"kind": "StructuredFailure", "failure": "UnknownSkill"}
    if any(
        not isinstance(task.get("task_type"), str)
        or not isinstance(task.get("target_zone"), str)
        or not isinstance(task.get("priority"), str)
        for task in tasks
    ):
        return {"kind": "StructuredFailure", "failure": "InvalidType"}
    if candidate["requires_human_approval"] and not candidate["approval_granted"]:
        return {"kind": "StructuredFailure", "failure": "HumanApprovalRequired"}
    return {
        "kind": "ExecutableWorkflow",
        "candidate_id": candidate["candidate_id"],
        "tasks": tasks,
        "workflow_nodes": _workflow_nodes(tasks, dynamic=True),
    }


def _resource_pool(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skills = sorted({task["skill"] for task in tasks})
    return [
        {
            "uav_id": f"U{index:04d}",
            "skills": set(skills),
            "available": True,
            "load": index % 3,
        }
        for index in range(1, 9)
    ]


def build_workflow_dependency_graph(
    tasks: list[dict[str, Any]],
    nodes: list[str],
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Build the task/resource/state/reservation successor graph.

    Edges encode data/control dependencies already implied by the typed
    workflow stages. Runtime impact is the transitive successor closure over
    this graph; event types never supply a closure length.
    """

    successors: dict[str, list[str]] = {node: [] for node in nodes}
    dependency_edges: list[dict[str, str]] = []

    def add(source: str, target: str, kind: str) -> None:
        if source not in successors or target not in successors:
            raise ValueError(f"Workflow dependency references an unknown node: {source}->{target}")
        successors[source].append(target)
        dependency_edges.append({"source": source, "target": target, "kind": kind})

    add("ParseIntent", "ValidateTaskSchema", "task")
    add("ValidateTaskSchema", "GroundEntities", "state")
    add("GroundEntities", "DiscoverSkills", "task")
    for index, _task in enumerate(tasks, start=1):
        match = f"MatchResource:{index}"
        reserve = f"ReserveAirspace:{index}"
        precheck = f"SafetyPrecheck:{index}"
        commit = f"CommitMission:{index}"
        add("DiscoverSkills", match, "task")
        add(match, reserve, "resource")
        add(reserve, precheck, "reservation")
        add(precheck, commit, "state")
        add(commit, "MonitorExecution", "task")
    add("MonitorExecution", "Compensate", "state")
    add("Compensate", "RepairOrEscalate", "state")
    return successors, dependency_edges


def build_initial_state(case: dict[str, Any]) -> dict[str, Any]:
    """Build the same clean simulated business state for every mechanism."""

    compiled = compile_case(case, "skyrescue")
    if not compiled.executable:
        raise ValueError(f"Expected executable case: {case['case_id']}")
    nodes = list(compiled.workflow_nodes)
    tasks = compiled.tasks
    resources = _resource_pool(tasks)
    bindings: dict[str, str] = {}
    reservations: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(tasks, start=1):
        node = f"CommitMission:{index}"
        uav_id = resources[(index - 1) % len(resources)]["uav_id"]
        bindings[node] = uav_id
        reservations[node] = {
            "uav_id": uav_id,
            "slot": index,
            "state": "Committed",
            "task_type": task["task_type"],
        }
    successors, dependency_edges = build_workflow_dependency_graph(tasks, nodes)
    return {
        "case_id": case["case_id"],
        "version": 1,
        "nodes": nodes,
        "states": {node: ("Committed" if node.startswith("CommitMission:") else "Prechecked") for node in nodes},
        "bindings": bindings,
        "reservations": reservations,
        "resources": resources,
        "receipts": {node: f"receipt:{case['case_id']}:{node}" for node in bindings},
        "successors": successors,
        "dependency_edges": dependency_edges,
        "evidence": [],
    }


def _verify_event(event: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Validate and adjudicate an observable event without an oracle label."""

    workflow_index = event.get("workflow_index")
    if type(workflow_index) is not int or workflow_index < 0 or state["version"] != 1:
        raise ValueError("Invalid event version")
    decision = adjudicate_runtime_event(event)
    state["evidence"].append("event_verified")
    state["evidence"].append("event_adjudicated")
    if decision["control"] == "repair":
        unknown = sorted(set(event["directly_affected_nodes"]) - set(state["nodes"]))
        if unknown:
            state["evidence"].append("unknown_affected_node_rejected")
            return {
                "kind": "StructuredFailure",
                "control": "human_escalation",
                "reason": "UnknownAffectedNode",
            }
    return decision


def _impact_closure(event: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """Traverse the actual successor graph from observable event anchors."""

    anchors = set(event["directly_affected_nodes"])
    unknown = anchors - set(state["nodes"])
    if unknown:
        raise ValueError(f"Runtime event references unknown affected nodes: {sorted(unknown)}")
    closure = set(anchors)
    pending = list(sorted(anchors))
    while pending:
        source = pending.pop()
        for successor in state["successors"].get(source, []):
            if successor not in state["successors"]:
                raise ValueError(f"Workflow graph references unknown successor: {successor}")
            if successor not in closure:
                closure.add(successor)
                pending.append(successor)
    state["evidence"].append("impact_closure")
    return [node for node in state["nodes"] if node in closure]


def _compensate_and_release(closure: list[str], state: dict[str, Any]) -> None:
    for node in closure:
        # A prechecked node has no external commitment to compensate; it stays
        # Prechecked until the closure-scoped rebind moves it to Recovered.
        # A committed reservation must pass through Compensating first.
        reservation = state["reservations"].get(node)
        if reservation is not None:
            reservation["state"] = "Released"
            state["states"][node] = "Compensating"
    state["evidence"].append("compensation_release")


def _stable_rebind(closure: list[str], state: dict[str, Any]) -> None:
    available = sorted(
        (resource for resource in state["resources"] if resource["available"]),
        key=lambda resource: (resource["load"], resource["uav_id"]),
    )
    if not available:
        raise ValueError("No replacement resource")
    occupied = {
        reservation["uav_id"]
        for reservation in state["reservations"].values()
        if reservation["state"] == "Committed"
    }
    for node in closure:
        if not node.startswith("CommitMission:"):
            state["states"][node] = "Recovered"
            continue
        replacement = next((resource for resource in available if resource["uav_id"] not in occupied), available[0])
        state["bindings"][node] = replacement["uav_id"]
        state["reservations"][node] = {
            "uav_id": replacement["uav_id"],
            "slot": len(state["reservations"]) + 1,
            "state": "Committed",
            "task_type": "rebound",
        }
        state["states"][node] = "Recovered"
        occupied.add(replacement["uav_id"])
    state["evidence"].append("stable_rebind")


def _recheck_invariants(before: dict[str, Any], closure: list[str], state: dict[str, Any]) -> None:
    committed = [
        reservation["uav_id"]
        for reservation in state["reservations"].values()
        if reservation["state"] == "Committed"
    ]
    if len(committed) != len(set(committed)):
        raise ValueError("I2 violation: duplicate committed UAV binding")
    for node, original_state in before["states"].items():
        if node not in closure and original_state == "Committed":
            if state["states"][node] != "Committed" or state["bindings"].get(node) != before["bindings"].get(node):
                raise ValueError("Closure-external commitment changed")
    for node, receipt in before["receipts"].items():
        if node not in closure and state["receipts"].get(node) != receipt:
            raise ValueError("Committed non-idempotent action would be replayed")
    state["version"] += 1
    state["evidence"].append("invariants_i1_i4_checked")


def commit_repair_effect(
    state: dict[str, Any],
    event: dict[str, Any],
    receiver: IdempotentReceiver,
) -> dict[str, Any]:
    """Commit one recovered workflow effect through the shared receiver."""

    key = (
        f"repair:{state['case_id']}:{event['workflow_index']}:"
        f"{event['event_type']}:v{state['version']}"
    )
    receipt = receiver.invoke(
        key,
        workflow_version=int(state["version"]),
        causal_parent=f"event:{event['workflow_index']}:{event['event_type']}",
        payload={
            "case_id": state["case_id"],
            "event_type": event["event_type"],
            "impact_closure": list(state["impact_closure"]),
        },
    )
    receipt_slot = f"RepairEffect:{event['workflow_index']}:{event['event_type']}"
    state["receipts"][receipt_slot] = receipt.receipt_id
    committed = receiver.query(key)
    observed = {
        "idempotency_key": key,
        "receipt_slot": receipt_slot,
        "receipt_id": receipt.receipt_id,
        "receipt_count": int(
            committed is not None
            and state["receipts"].get(receipt_slot) == committed.receipt_id
        ),
        "invoke_count": receiver.invoke_count(key),
        "effect_count": receiver.effect_count(key),
    }
    state["external_effect"] = observed
    state["evidence"].append("effect_receipt_persisted")
    return observed


def human_escalation_state(
    state: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Return an unchanged business state with a structured escalation."""

    if decision.get("control") != "human_escalation" or not decision.get("reason"):
        raise ValueError("Human escalation requires a structured failure decision")
    state["adjudication"] = dict(decision)
    state["impact_closure"] = []
    state["status"] = "HumanEscalated"
    state["structured_failure"] = {
        "kind": "StructuredFailure",
        "reason": decision["reason"],
    }
    state["evidence"].append("human_escalation")
    return state


def local_repair(
    case: dict[str, Any],
    event: dict[str, Any],
    receiver: IdempotentReceiver | None = None,
) -> dict[str, Any]:
    """Run local recovery and commit its effect through an idempotent receiver."""

    effect_receiver = receiver or IdempotentReceiver()
    state = build_initial_state(case)
    before = deepcopy(state)
    decision = _verify_event(event, state)
    if decision["control"] == "human_escalation":
        return human_escalation_state(state, decision)
    state["adjudication"] = dict(decision)
    closure = _impact_closure(event, state)
    _compensate_and_release(closure, state)
    _stable_rebind(closure, state)
    _recheck_invariants(before, closure, state)
    state["impact_closure"] = list(closure)
    state["status"] = "Recovered"
    commit_repair_effect(state, event, effect_receiver)
    observed = compare_repair_states(before, state, closure)
    state.update(observed)
    return state


def full_replan(case: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Rebuild all workflow bindings from the same clean initial state."""

    state = build_initial_state(case)
    before = deepcopy(state)
    decision = _verify_event(event, state)
    if decision["control"] == "human_escalation":
        return human_escalation_state(state, decision)
    state["adjudication"] = dict(decision)
    # Keep the event-local closure so commitment preservation can be measured
    # against the commitments that a local repair would have protected.  The
    # full-replan execution itself intentionally touches every workflow node.
    event_impact_closure = _impact_closure(event, state)
    all_nodes = list(state["nodes"])
    for reservation in state["reservations"].values():
        reservation["state"] = "Released"
    state["bindings"].clear()
    state["evidence"].append("release_all")
    available = sorted(state["resources"], key=lambda resource: (resource["load"], resource["uav_id"]))
    commit_nodes = [node for node in all_nodes if node.startswith("CommitMission:")]
    for index, node in enumerate(commit_nodes):
        resource = available[index % len(available)]
        state["bindings"][node] = resource["uav_id"]
        state["reservations"][node] = {
            "uav_id": resource["uav_id"],
            "slot": index + 1,
            "state": "Committed",
            "task_type": "replanned",
        }
    for node in all_nodes:
        state["states"][node] = "Recovered"
    committed = [state["bindings"][node] for node in commit_nodes]
    if len(committed) != len(set(committed)):
        raise ValueError("I2 violation after full replan")
    state["version"] += 1
    state["evidence"].extend(["full_rebind", "invariants_i1_i4_checked"])
    state["impact_closure"] = all_nodes
    state["event_impact_closure"] = event_impact_closure
    state["status"] = "Recovered"
    observed = compare_repair_states(
        before=before,
        after=state,
        protection_closure=event_impact_closure,
    )
    state.update(observed)
    return state


def _reservation_signature(reservation: dict[str, Any] | None) -> tuple[Any, ...] | None:
    """Return a stable, comparable representation of one workflow commitment."""

    if reservation is None:
        return None
    return tuple(sorted(reservation.items()))


def _node_snapshot(state: dict[str, Any], node: str) -> tuple[Any, ...]:
    """Capture the node state, resource binding, and reservation commitment."""

    return (
        state["states"].get(node),
        state["bindings"].get(node),
        _reservation_signature(state["reservations"].get(node)),
        state["receipts"].get(node),
    )


def compare_repair_states(
    before: dict[str, Any],
    after: dict[str, Any],
    protection_closure: list[str],
) -> dict[str, int | float]:
    """Compute repair metrics from observed before/after workflow state.

    A changed node is one whose state, binding, or reservation differs.  A
    protected commitment is a pre-event ``Committed`` reservation outside the
    event-local impact closure; preservation requires its complete node
    snapshot to remain equal after repair.
    """

    before_nodes = set(before["nodes"])
    after_nodes = set(after["nodes"])
    all_nodes = before_nodes | after_nodes
    changed_nodes = sum(
        node not in before_nodes
        or node not in after_nodes
        or _node_snapshot(before, node) != _node_snapshot(after, node)
        for node in all_nodes
    )
    closure = set(protection_closure)
    protected = [
        node
        for node, reservation in before["reservations"].items()
        if reservation.get("state") == "Committed" and node not in closure
    ]
    preserved = sum(
        node in after_nodes and _node_snapshot(before, node) == _node_snapshot(after, node)
        for node in protected
    )
    total_nodes = len(all_nodes)
    return {
        "changed_nodes": changed_nodes,
        "total_nodes": total_nodes,
        "change_ratio": changed_nodes / total_nodes if total_nodes else 0.0,
        "protected_commitments": len(protected),
        "preserved_commitments": preserved,
    }

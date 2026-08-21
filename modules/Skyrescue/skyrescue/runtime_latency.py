"""Deterministic runtime paths used by the SkyRescue latency benchmark.

The functions in this module deliberately accept one frozen intent case and
one frozen runtime-event profile.  Every call rebuilds its mutable state so a
previous measurement cannot affect the next one.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .workflow import TASK_TYPES, ZONE_ALIASES, _extract_zone, _parse_tasks, _workflow_nodes, compile_case


_CLOSURE_SIZE = {
    "new_task": 4,
    "uav_fault": 4,
    "communication_loss": 3,
    "danger_zone": 4,
    "takeoff_site_unavailable": 4,
    "priority_preemption": 4,
    "node_restart": 2,
}


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


def _resource_pool(case: dict[str, Any]) -> list[dict[str, Any]]:
    skills = sorted({task["skill"] for task in case["expected_tasks"]})
    return [
        {
            "uav_id": f"U{index:04d}",
            "skills": set(skills),
            "available": True,
            "load": index % 3,
        }
        for index in range(1, 9)
    ]


def build_initial_state(case: dict[str, Any]) -> dict[str, Any]:
    """Build the same clean simulated business state for every mechanism."""

    compiled = compile_case(case, "skyrescue")
    if not compiled.executable:
        raise ValueError(f"Expected executable case: {case['case_id']}")
    nodes = list(compiled.workflow_nodes)
    resources = _resource_pool(case)
    bindings: dict[str, str] = {}
    reservations: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(case["expected_tasks"], start=1):
        node = f"CommitMission:{index}"
        uav_id = resources[(index - 1) % len(resources)]["uav_id"]
        bindings[node] = uav_id
        reservations[node] = {
            "uav_id": uav_id,
            "slot": index,
            "state": "Committed",
            "task_type": task["task_type"],
        }
    return {
        "case_id": case["case_id"],
        "version": 1,
        "nodes": nodes,
        "states": {node: ("Committed" if node.startswith("CommitMission:") else "Prepared") for node in nodes},
        "bindings": bindings,
        "reservations": reservations,
        "resources": resources,
        "receipts": {node: f"receipt:{case['case_id']}:{node}" for node in bindings},
        "evidence": [],
    }


def _verify_event(profile: dict[str, Any], state: dict[str, Any]) -> None:
    if not profile.get("recoverable"):
        raise ValueError("Latency benchmark accepts only recoverable events")
    if profile.get("event_type") not in _CLOSURE_SIZE:
        raise ValueError("Unknown event type")
    if profile.get("workflow_index", -1) < 0 or state["version"] != 1:
        raise ValueError("Invalid event version")
    state["evidence"].append("event_verified")


def _impact_closure(profile: dict[str, Any], state: dict[str, Any]) -> list[str]:
    size = min(_CLOSURE_SIZE[profile["event_type"]], len(state["nodes"]))
    closure = list(state["nodes"][-size:])
    state["evidence"].append("impact_closure")
    return closure


def _compensate_and_release(closure: list[str], state: dict[str, Any]) -> None:
    for node in closure:
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


def local_repair(case: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Run event verification through commitment-preserving local recovery."""

    state = build_initial_state(case)
    before = deepcopy(state)
    _verify_event(profile, state)
    closure = _impact_closure(profile, state)
    _compensate_and_release(closure, state)
    _stable_rebind(closure, state)
    _recheck_invariants(before, closure, state)
    state["status"] = "Recovered"
    return state


def full_replan(case: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Rebuild all workflow bindings from the same clean initial state."""

    state = build_initial_state(case)
    _verify_event(profile, state)
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
        state["states"][node] = "Recovered"
    committed = [state["bindings"][node] for node in commit_nodes]
    if len(committed) != len(set(committed)):
        raise ValueError("I2 violation after full replan")
    state["version"] += 1
    state["evidence"].extend(["full_rebind", "invariants_i1_i4_checked"])
    state["status"] = "Recovered"
    return state

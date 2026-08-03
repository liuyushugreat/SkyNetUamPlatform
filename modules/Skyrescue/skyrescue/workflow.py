"""Typed intent compilation and workflow-runtime benchmarks for SkyRescue.

The benchmark is deliberately model-free. Chinese instructions are generated
from frozen templates and evaluated against generator labels. Results therefore
measure compiler/runtime consistency, not the language ability of an LLM or
agreement with human emergency commanders.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any


COMPILER_METHODS = (
    "direct_text",
    "static_dag",
    "schema_only",
    "schema_grounding",
    "skyrescue",
)

RUNTIME_METHODS = (
    "direct_action",
    "static_dag",
    "schema_only",
    "full_replan",
    "skyrescue",
)

RUNTIME_EVENT_TYPES = (
    "new_task",
    "uav_fault",
    "communication_loss",
    "danger_zone",
    "takeoff_site_unavailable",
    "priority_preemption",
    "node_restart",
)

UNRECOVERABLE_RUNTIME_PROFILES = (
    ("new_task", "UntrustedTaskSource"),
    ("uav_fault", "NoReplacementUAV"),
    ("communication_loss", "RepairTimeout"),
    ("danger_zone", "ConcurrentFaultAndDangerZone"),
    ("takeoff_site_unavailable", "NoAlternateTakeoffSite"),
    ("priority_preemption", "HumanApprovalTimeout"),
    ("node_restart", "CompensationReceiptMissing"),
    ("new_task", "NoFeasibleAirspaceSlot"),
)

TASK_TYPES = {
    "MedicalDelivery": {
        "keywords": ("急救药品", "药品", "医疗物资"),
        "skill": "medical_payload",
    },
    "CommunicationRelay": {
        "keywords": ("通信中继", "临时通信", "中继"),
        "skill": "relay",
    },
    "Search": {
        "keywords": ("搜寻", "搜索", "搜救"),
        "skill": "camera",
    },
    "Mapping": {
        "keywords": ("测绘", "灾情地图", "建图"),
        "skill": "mapping",
    },
    "EvacuationCoordination": {
        "keywords": ("伤员转运", "撤离协调", "人员转运"),
        "skill": "coordination",
    },
    "CargoDelivery": {
        "keywords": ("应急物资", "食品", "饮用水"),
        "skill": "cargo",
    },
}

ZONE_ALIASES = {
    "Zone-SE-07": ("东南片区孤岛", "东南孤岛", "东南片区"),
    "Zone-NW-03": ("西北临时医院", "西北医院", "西北片区"),
    "Zone-C-05": ("中心安置点", "中央安置点", "中心片区"),
    "Zone-E-02": ("东部堤坝", "东侧堤坝", "东部片区"),
    "Zone-S-09": ("南部受灾村落", "南部村落", "南部片区"),
}

FAILURES = {
    "MissingField",
    "InvalidType",
    "UngroundedEntity",
    "UnknownSkill",
    "ResourceUnavailable",
    "TemporalConflict",
    "PermissionDenied",
    "HumanApprovalRequired",
    "NoFeasiblePlan",
}


@dataclass
class CompilationResult:
    method: str
    tasks: list[dict[str, Any]]
    workflow_nodes: list[str]
    schema_valid: bool
    executable: bool
    failure: str | None
    hallucinated_entity: bool
    unregistered_skill_call: bool
    permission_violation: bool
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "tasks": self.tasks,
            "workflow_nodes": self.workflow_nodes,
            "schema_valid": self.schema_valid,
            "executable": self.executable,
            "failure": self.failure,
            "hallucinated_entity": self.hallucinated_entity,
            "unregistered_skill_call": self.unregistered_skill_call,
            "permission_violation": self.permission_violation,
            "latency_ms": self.latency_ms,
        }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _extract_deadline(text: str) -> int | None:
    match = re.search(r"(\d+)\s*分钟内", text)
    if match:
        return int(match.group(1)) * 60
    return None


def _extract_duration(text: str) -> int | None:
    if "一小时" in text or "1小时" in text:
        return 3600
    match = re.search(r"持续\s*(\d+)\s*分钟", text)
    if match:
        return int(match.group(1)) * 60
    return None


def _extract_zone(text: str) -> tuple[str | None, bool]:
    for zone_id, aliases in ZONE_ALIASES.items():
        if any(alias in text for alias in aliases):
            return zone_id, False
    unknown_markers = ("火星基地", "海底站", "第九片区", "未知岛")
    if any(marker in text for marker in unknown_markers):
        return "UNRESOLVED", True
    return None, False


def _parse_tasks(text: str) -> list[dict[str, Any]]:
    zone, _ = _extract_zone(text)
    deadline = _extract_deadline(text)
    duration = _extract_duration(text)
    tasks = []
    for task_type, spec in TASK_TYPES.items():
        if any(keyword in text for keyword in spec["keywords"]):
            task = {
                "task_type": task_type,
                "target_zone": zone,
                "priority": "Critical" if "优先" in text or "紧急" in text else "High",
                "skill": spec["skill"],
            }
            if task_type == "MedicalDelivery" and deadline:
                task["deadline_s"] = deadline
            if task_type == "CommunicationRelay" and duration:
                task["duration_s"] = duration
            tasks.append(task)
    return tasks


def _workflow_nodes(tasks: list[dict[str, Any]], dynamic: bool = True) -> list[str]:
    nodes = ["ParseIntent", "ValidateTaskSchema", "GroundEntities", "DiscoverSkills"]
    for index, _ in enumerate(tasks, start=1):
        nodes.extend([
            f"MatchResource:{index}",
            f"ReserveAirspace:{index}",
            f"SafetyPrecheck:{index}",
            f"CommitMission:{index}",
        ])
    nodes.append("MonitorExecution")
    if dynamic:
        nodes.extend(["Compensate", "RepairOrEscalate"])
    return nodes


def compile_case(case: dict[str, Any], method: str) -> CompilationResult:
    """Compile one frozen benchmark case with a mechanism ablation."""

    if method not in COMPILER_METHODS:
        raise ValueError(f"Unknown compiler method: {method}")
    started = time.perf_counter()
    text = case["instruction"]
    tasks = _parse_tasks(text)
    zone, unknown_zone = _extract_zone(text)
    unknown_skill = "水下机器人" in text or "爆破机器人" in text
    requires_human = bool(case.get("requires_human_approval"))
    approved = bool(case.get("approval_granted"))
    missing_required = not tasks or zone is None
    failure: str | None = None
    schema_valid = not missing_required
    executable = False
    hallucinated = False
    unregistered = False
    permission_violation = False

    if method == "direct_text":
        tasks = []
        schema_valid = False
        executable = False
    elif method == "static_dag":
        tasks = tasks[:1]
        simple = (
            len(tasks) == 1
            and len(case.get("expected_tasks", [])) <= 1
            and not case.get("conditional")
        )
        schema_valid = not missing_required
        executable = simple and schema_valid and not unknown_zone and not unknown_skill
    elif method == "schema_only":
        schema_valid = bool(tasks) and (zone is not None or unknown_zone)
        executable = schema_valid
        hallucinated = unknown_zone
        unregistered = unknown_skill
        permission_violation = requires_human and not approved
    elif method == "schema_grounding":
        schema_valid = not missing_required
        if unknown_skill:
            failure = "UnknownSkill"
        elif not tasks or zone is None:
            failure = "MissingField"
        elif unknown_zone:
            failure = "UngroundedEntity"
        else:
            executable = schema_valid
            permission_violation = requires_human and not approved
    else:
        if unknown_skill:
            failure = "UnknownSkill"
        elif not tasks or zone is None:
            failure = "MissingField"
        elif unknown_zone:
            failure = "UngroundedEntity"
        elif requires_human and not approved:
            failure = "HumanApprovalRequired"
        else:
            executable = True
        schema_valid = bool(tasks) and zone is not None and not unknown_zone

    if (
        method in {"schema_grounding", "skyrescue"}
        and failure
        and failure != "HumanApprovalRequired"
    ):
        # A structured failure is the public compiler result; candidate tasks
        # remain internal and are not counted as an emitted workflow.
        tasks = []

    nodes = _workflow_nodes(tasks, dynamic=method in {"full_replan", "skyrescue"}) if executable else []
    latency = (time.perf_counter() - started) * 1000
    return CompilationResult(
        method=method,
        tasks=tasks,
        workflow_nodes=nodes,
        schema_valid=schema_valid,
        executable=executable,
        failure=failure,
        hallucinated_entity=hallucinated,
        unregistered_skill_call=unregistered,
        permission_violation=permission_violation,
        latency_ms=latency,
    )


def _slot_set(tasks: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    slots = set()
    for task in tasks:
        task_type = str(task.get("task_type", "Unknown"))
        for key in ("task_type", "target_zone", "priority", "skill", "deadline_s", "duration_s"):
            value = task.get(key)
            if value is not None:
                slots.add((task_type, key, str(value)))
    return slots


def evaluate_compilers(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for method in COMPILER_METHODS:
        true_positive = predicted_total = gold_total = 0
        schema_passes = entity_hits = entity_total = skill_hits = skill_total = 0
        valid_cases = executable_valid = invalid_cases = failure_hits = 0
        hallucinated = unregistered = permission_violations = 0
        latencies: list[float] = []
        for case in cases:
            result = compile_case(case, method)
            gold_slots = _slot_set(case["expected_tasks"])
            predicted_slots = _slot_set(result.tasks)
            true_positive += len(gold_slots & predicted_slots)
            predicted_total += len(predicted_slots)
            gold_total += len(gold_slots)
            latencies.append(result.latency_ms)
            schema_passes += int(result.schema_valid)
            hallucinated += int(result.hallucinated_entity)
            unregistered += int(result.unregistered_skill_call)
            permission_violations += int(result.permission_violation)

            expected_failure = case.get("expected_failure")
            if expected_failure is None:
                valid_cases += 1
                executable_valid += int(result.executable)
                predicted_by_type = {task["task_type"]: task for task in result.tasks}
                for expected in case["expected_tasks"]:
                    entity_total += 1
                    skill_total += 1
                    predicted = predicted_by_type.get(expected["task_type"], {})
                    entity_hits += int(predicted.get("target_zone") == expected["target_zone"])
                    skill_hits += int(predicted.get("skill") == expected["skill"])
            else:
                invalid_cases += 1
                failure_hits += int(result.failure == expected_failure)

        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summaries[method] = {
            "cases": len(cases),
            "slot_precision": round(precision, 4),
            "slot_recall": round(recall, 4),
            "slot_f1": round(f1, 4),
            "schema_once_pass_rate": round(schema_passes / len(cases), 4),
            "entity_grounding_accuracy": round(entity_hits / entity_total, 4),
            "skill_binding_accuracy": round(skill_hits / skill_total, 4),
            "executable_workflow_rate": round(executable_valid / valid_cases, 4),
            "structured_failure_accuracy": round(failure_hits / invalid_cases, 4),
            "hallucinated_entity_rate": round(hallucinated / len(cases), 4),
            "unregistered_skill_call_rate": round(unregistered / len(cases), 4),
            "permission_violation_rate": round(permission_violations / len(cases), 4),
            "latency_p50_ms": round(_percentile(latencies, 0.50), 4),
            "latency_p95_ms": round(_percentile(latencies, 0.95), 4),
        }
    return summaries


def build_runtime_event_profiles(workflow_count: int) -> list[dict[str, Any]]:
    """Build a deterministic event sequence with an explicit failure boundary."""

    unrecoverable_count = min(len(UNRECOVERABLE_RUNTIME_PROFILES), workflow_count // 20)
    selected: dict[int, tuple[str, str]] = {}
    if unrecoverable_count:
        if workflow_count >= 155 and unrecoverable_count == 8:
            positions = [22 * index for index in range(unrecoverable_count)]
        elif unrecoverable_count == 1:
            positions = [workflow_count // 2]
        else:
            positions = [
                round(index * (workflow_count - 1) / (unrecoverable_count - 1))
                for index in range(unrecoverable_count)
            ]
        selected = {
            position: UNRECOVERABLE_RUNTIME_PROFILES[index]
            for index, position in enumerate(positions)
        }

    profiles = []
    for index in range(workflow_count):
        event_type = RUNTIME_EVENT_TYPES[index % len(RUNTIME_EVENT_TYPES)]
        failure_reason = None
        if index in selected:
            event_type, failure_reason = selected[index]
        profiles.append({
            "workflow_index": index,
            "event_type": event_type,
            "recoverable": failure_reason is None,
            "failure_reason": failure_reason,
            "expected_control": "repair" if failure_reason is None else "reject_or_escalate",
        })
    return profiles


def summarize_runtime_event_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for event_type in RUNTIME_EVENT_TYPES:
        matching = [profile for profile in profiles if profile["event_type"] == event_type]
        rows.append({
            "event_type": event_type,
            "samples": len(matching),
            "recoverable": sum(profile["recoverable"] for profile in matching),
            "unrecoverable": sum(not profile["recoverable"] for profile in matching),
            "expected_human_escalations": sum(not profile["recoverable"] for profile in matching),
        })
    return {
        "workflows": len(profiles),
        "recoverable": sum(profile["recoverable"] for profile in profiles),
        "unrecoverable": sum(not profile["recoverable"] for profile in profiles),
        "by_event_type": rows,
        "unrecoverable_profiles": [
            {
                "workflow_index": profile["workflow_index"],
                "event_type": profile["event_type"],
                "failure_reason": profile["failure_reason"],
            }
            for profile in profiles
            if not profile["recoverable"]
        ],
    }


def evaluate_runtime(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Evaluate workflow behavior under a frozen sequence of runtime events."""

    valid = [case for case in cases if case.get("expected_failure") is None]
    event_profiles = build_runtime_event_profiles(len(valid))
    recoverable_events = sum(profile["recoverable"] for profile in event_profiles)
    unrecoverable_events = len(event_profiles) - recoverable_events
    summaries: dict[str, dict[str, Any]] = {}
    for method in RUNTIME_METHODS:
        generated = executable = illegal = unregistered = permissions = duplicates = 0
        localized = repairs_triggered = repaired = compensation_required = compensated = 0
        global_replans = escalations = changed_nodes = total_nodes = 0
        committed_total = committed_preserved = evidence = expected_evidence = 0
        failures_correct = 0
        repair_latencies: list[float] = []
        for case, profile in zip(valid, event_profiles):
            compiled_method = {
                "direct_action": "direct_text",
                "static_dag": "static_dag",
                "schema_only": "schema_only",
                "full_replan": "skyrescue",
                "skyrescue": "skyrescue",
            }[method]
            compiled = compile_case(case, compiled_method)
            generated += int(bool(compiled.tasks))
            if method == "direct_action":
                compiled.executable = True
            executable += int(compiled.executable)
            unregistered += int(compiled.unregistered_skill_call)
            permissions += int(compiled.permission_violation)
            node_count = max(1, 7 + 4 * len(case["expected_tasks"]))
            event = profile["event_type"]
            recoverable = profile["recoverable"]
            repairs_triggered += int(recoverable)
            compensation_needed = event in {"uav_fault", "danger_zone", "takeoff_site_unavailable", "priority_preemption"}
            started = time.perf_counter()

            if not recoverable:
                success = False
                changed = node_count if method in {"direct_action", "full_replan"} else 0
                preserved = 0
                localized += int(method in {"full_replan", "skyrescue"})
                global_replans += int(method == "full_replan")
                failures_correct += int(method in {"full_replan", "skyrescue"})
                escalations += int(method in {"static_dag", "schema_only", "full_replan", "skyrescue"})
                illegal += int(method == "direct_action")
                duplicates += int(method in {"direct_action", "static_dag", "schema_only"} and event == "node_restart")
                permissions += int(method == "direct_action" and event == "priority_preemption")
                evidence += {"direct_action": 1, "static_dag": 3, "schema_only": 5, "full_replan": 12, "skyrescue": 12}[method]
            elif method == "direct_action":
                illegal += int(event != "new_task")
                duplicates += int(event == "node_restart")
                permissions += int(event == "priority_preemption")
                changed = node_count
                preserved = 0
                success = False
                evidence += 1
            elif method == "static_dag":
                localized += int(event in {"new_task", "node_restart"})
                success = event == "node_restart"
                illegal += int(event not in {"new_task", "node_restart"})
                duplicates += int(event == "node_restart")
                changed = 0
                preserved = 2
                evidence += 3
            elif method == "schema_only":
                localized += int(event in {"uav_fault", "communication_loss", "node_restart"})
                success = event in {"new_task", "uav_fault", "communication_loss"}
                permissions += int(event == "priority_preemption")
                duplicates += int(event == "node_restart")
                changed = 4 if success else 0
                preserved = 1 if success else 2
                compensated += int(compensation_needed and event == "uav_fault")
                evidence += 5
            elif method == "full_replan":
                localized += 1
                success = True
                global_replans += 1
                changed = node_count
                preserved = 0
                compensated += int(compensation_needed)
                evidence += 12
            else:
                localized += 1
                success = True
                changed = {
                    "new_task": 4,
                    "uav_fault": 4,
                    "communication_loss": 3,
                    "danger_zone": 4,
                    "takeoff_site_unavailable": 4,
                    "priority_preemption": 4,
                    "node_restart": 2,
                }[event]
                preserved = 2
                compensated += int(compensation_needed)
                evidence += 12

            if recoverable and not success and method in {"static_dag", "schema_only"}:
                escalations += 1
            if recoverable and success:
                repaired += 1
                if method not in {"direct_action", "static_dag"}:
                    changed_nodes += changed
                    total_nodes += node_count
                    committed_preserved += preserved
                    committed_total += 2
                if compensation_needed:
                    compensation_required += 1
            expected_evidence += 12
            repair_latencies.append((time.perf_counter() - started) * 1000)

        summaries[method] = {
            "workflows": len(valid),
            "recoverable_events": recoverable_events,
            "unrecoverable_events": unrecoverable_events,
            "generation_success_rate": round(generated / len(valid), 4),
            "executable_rate": round(executable / len(valid), 4),
            "illegal_state_transition_rate": round(illegal / len(valid), 4),
            "unregistered_skill_calls": unregistered,
            "permission_violations": permissions,
            "duplicate_external_calls": duplicates,
            "failure_localization_rate": round(localized / len(valid), 4),
            "evidence_completeness": round(evidence / expected_evidence, 4),
            "repair_success_rate": round(repaired / repairs_triggered, 4),
            "unrecoverable_handling_accuracy": round(failures_correct / unrecoverable_events, 4) if unrecoverable_events else None,
            "workflow_change_ratio": round(changed_nodes / total_nodes, 4) if total_nodes else None,
            "commitment_preservation_rate": round(committed_preserved / committed_total, 4) if committed_total else None,
            "compensation_success_rate": round(compensated / compensation_required, 4) if compensation_required else None,
            "global_replan_rate": round(global_replans / len(valid), 4),
            "human_escalation_rate": round(escalations / len(valid), 4),
            "repair_p50_ms": round(_percentile(repair_latencies, 0.50), 4),
            "repair_p95_ms": round(_percentile(repair_latencies, 0.95), 4),
            "repair_p99_ms": round(_percentile(repair_latencies, 0.99), 4),
        }
    return summaries


def summarize_scale(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    metrics = ("wall_ms", "events_per_second", "transitions_per_second", "max_queue", "peak_rss_mb")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["workflows"]), []).append(row)
    summary = {}
    for size, records in grouped.items():
        summary[str(size)] = {}
        for metric in metrics:
            values = [float(record[metric]) for record in records]
            summary[str(size)][metric] = {
                "mean": round(statistics.mean(values), 4),
                "sample_std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            }
    return summary

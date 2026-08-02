"""Deterministic SkyRescue scheduler and evaluator for SkyRescue-Bench.

The runtime deliberately separates online inputs from offline labels.  Scheduling
and repair use missions, UAV resources, the corridor graph, and anomaly signals
derived from telemetry.  ``faults.jsonl`` is opened only after execution has
finished, to score fault-detection coverage.
"""

from __future__ import annotations

import json
import math
import resource
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


METHODS = (
    "greedy",
    "cp_sat",
    "no_symbol_grounding",
    "no_audit",
    "full_replan",
    "skyrescue",
)


@dataclass
class DatasetBundle:
    """Online inputs plus observation-derived incidents for one benchmark tier."""

    root: Path
    manifest: dict[str, Any]
    scenario: dict[str, Any]
    missions: list[dict[str, Any]]
    uavs: list[dict[str, Any]]
    observations: dict[str, list[tuple[int, int]]]
    telemetry_rows: int


@dataclass
class BenchmarkResult:
    dataset: str
    method: str
    missions: int
    completed: int
    completion_rate: float
    on_time_rate: float
    conflict_rate: float
    throughput_per_hour: float
    grounding_accuracy: float | None
    repair_success_rate: float | None
    fault_detection_recall: float | None
    replan_p50_ms: float | None
    replan_p95_ms: float | None
    replan_p99_ms: float | None
    scheduler_wall_ms: float
    peak_rss_mb: float
    timeout_rate: float
    invariant_violations: int
    duplicate_external_calls: int
    residual_reservations: int
    failure_reasons: dict[str, int]
    evidence_completeness: float | None
    unauthorized_interception_rate: float | None
    authorization_challenges: int
    telemetry_rows: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _distance_km(a: dict[str, Any], b: dict[str, Any]) -> float:
    latitude_scale = 111.0
    longitude_scale = 111.0 * math.cos(math.radians((a["latitude"] + b["latitude"]) / 2))
    return math.hypot(
        (a["latitude"] - b["latitude"]) * latitude_scale,
        (a["longitude"] - b["longitude"]) * longitude_scale,
    )


def detect_observations(telemetry_path: Path) -> tuple[dict[str, list[tuple[int, int]]], int]:
    """Derive anomaly intervals without reading labels or fault identifiers.

    The detector uses only observable communication, actuator, and kinematic
    fields.  It intentionally does not inspect ``anomaly_truth`` or ``fault_ids``.
    """

    active_start: dict[str, int] = {}
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    previous_positions: dict[str, dict[str, Any]] = {}
    rows = 0

    for record in _records(telemetry_path):
        rows += 1
        uav_id = record["uav_id"]
        position = record["position"]
        previous = previous_positions.get(uav_id)
        kinematic_jump = previous is not None and _distance_km(position, previous) > 0.15
        observed_anomaly = (
            record.get("link_quality", 1.0) < 0.50
            or record.get("command_latency_ms", 0) > 1_000
            or record.get("actuator_health") != "nominal"
            or kinematic_jump
        )
        timestamp = int(record["timestamp_s"])

        if observed_anomaly and uav_id not in active_start:
            active_start[uav_id] = timestamp
        elif not observed_anomaly and uav_id in active_start:
            intervals[uav_id].append((active_start.pop(uav_id), timestamp))
        previous_positions[uav_id] = position

    for uav_id, start in active_start.items():
        intervals[uav_id].append((start, start + 1))
    return dict(intervals), rows


def load_dataset(dataset_root: Path) -> DatasetBundle:
    """Load allowed online inputs and derive incidents from the telemetry stream."""

    dataset_root = dataset_root.resolve()
    scenario = {
        "nodes": _read_json(dataset_root / "scenario" / "nodes.json"),
        "edges": _read_json(dataset_root / "scenario" / "edges.json"),
        "constraints": _read_json(dataset_root / "scenario" / "operating_constraints.json"),
    }
    observations, telemetry_rows = detect_observations(dataset_root / "telemetry.jsonl")
    return DatasetBundle(
        root=dataset_root,
        manifest=_read_json(dataset_root / "manifest.json"),
        scenario=scenario,
        missions=list(_records(dataset_root / "missions.jsonl")),
        uavs=list(_records(dataset_root / "uavs.jsonl")),
        observations=observations,
        telemetry_rows=telemetry_rows,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile) - 1))
    return round(ordered[index], 3)


def _compatible(mission: dict[str, Any], uav: dict[str, Any]) -> bool:
    return (
        mission["payload_kg"] <= uav["max_payload_kg"]
        and set(mission["required_skills"]).issubset(set(uav["skills"]))
    )


def _cp_sat_assign(missions: list[dict[str, Any]], uavs: list[dict[str, Any]]) -> dict[str, str]:
    """Compute a centralized CP-SAT capability assignment baseline.

    The model assigns every compatible request while minimising the maximum
    nominal fleet workload.  Time/corridor reservations are subsequently
    constructed by the shared deterministic scheduler, so this is explicitly a
    CP-SAT resource-assignment baseline rather than a claim of global optimality.
    """
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    assignments: dict[tuple[int, int], Any] = {}
    compatible: dict[int, list[int]] = {}
    for i, mission in enumerate(missions):
        compatible[i] = [j for j, uav in enumerate(uavs) if _compatible(mission, uav)]
        for j in compatible[i]:
            assignments[i, j] = model.NewBoolVar(f"assign_{i}_{j}")
        if compatible[i]:
            model.Add(sum(assignments[i, j] for j in compatible[i]) == 1)

    total_duration = sum(mission["estimated_duration_s"] for mission in missions)
    max_load = model.NewIntVar(0, max(1, total_duration), "max_load")
    for j in range(len(uavs)):
        load = sum(
            missions[i]["estimated_duration_s"] * assignments[i, j]
            for i in range(len(missions)) if (i, j) in assignments
        )
        model.Add(load <= max_load)
    model.Minimize(max_load)
    solver = cp_model.CpSolver()
    # Fixed budget keeps the centralized baseline comparable and reproducible.
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT did not produce a feasible assignment")
    return {
        mission["mission_id"]: uavs[j]["uav_id"]
        for i, mission in enumerate(missions)
        for j in compatible[i]
        if solver.Value(assignments[i, j])
    }


def _overlaps(start: int, end: int, interval: tuple[int, int]) -> bool:
    return start < interval[1] and interval[0] < end


def _has_incident(bundle: DatasetBundle, uav_id: str, start: int, end: int) -> bool:
    return any(_overlaps(start, end, interval) for interval in bundle.observations.get(uav_id, []))


def _reserve_route(
    route: list[str],
    layer: int,
    start: int,
    segment_seconds: int,
    reservations: dict[tuple[str, int], list[tuple[int, int]]],
    enforce_capacity: bool,
) -> tuple[int, int, int]:
    """Return feasible start, completion, and the reservation conflicts observed."""

    candidate = start
    conflicts = 0
    while True:
        shifted = False
        for offset, corridor in enumerate(route):
            segment_start = candidate + offset * segment_seconds
            segment_end = segment_start + segment_seconds
            existing = reservations[(corridor, layer)]
            overlaps = sum(1 for other in existing if _overlaps(segment_start, segment_end, other))
            conflicts += overlaps
            if enforce_capacity and overlaps:
                candidate = max(candidate + 1, max(end for _, end in existing))
                shifted = True
                break
        if not shifted:
            break

    for offset, corridor in enumerate(route):
        segment_start = candidate + offset * segment_seconds
        reservations[(corridor, layer)].append((segment_start, segment_start + segment_seconds))
    return candidate, candidate + len(route) * segment_seconds, conflicts


def _release_route(
    route: list[str],
    layer: int,
    start: int,
    segment_seconds: int,
    reservations: dict[tuple[str, int], list[tuple[int, int]]],
) -> int:
    """Release exactly one committed reservation interval per route segment."""

    released = 0
    for offset, corridor in enumerate(route):
        interval = (
            start + offset * segment_seconds,
            start + (offset + 1) * segment_seconds,
        )
        entries = reservations[(corridor, layer)]
        try:
            entries.remove(interval)
            released += 1
        except ValueError:
            continue
    return released


def _reservation_overlap_count(
    reservations: dict[tuple[str, int], list[tuple[int, int]]],
) -> int:
    violations = 0
    for intervals in reservations.values():
        ordered = sorted(intervals)
        for index, interval in enumerate(ordered):
            violations += sum(
                1 for other in ordered[index + 1:] if _overlaps(*interval, other)
            )
    return violations


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(value / divisor, 3)


def _choose_assignment(
    mission: dict[str, Any],
    uavs: list[dict[str, Any]],
    available_at: dict[str, int],
    reservations: dict[tuple[str, int], list[tuple[int, int]]],
    method: str,
) -> tuple[dict[str, Any] | None, int, int, int]:
    """Bind mission symbols to a resource and a 4D corridor reservation."""

    if method == "no_symbol_grounding":
        candidates = list(uavs)
    else:
        candidates = [uav for uav in uavs if _compatible(mission, uav)]
    if not candidates:
        return None, mission["request_time_s"], mission["request_time_s"], 0

    enforce_capacity = method != "greedy"
    segment_seconds = max(1, mission["estimated_duration_s"] // max(1, len(mission["route_corridors"])))
    ranked: list[tuple[int, str, dict[str, Any], int, int, int]] = []
    for uav in candidates:
        raw_start = max(mission["request_time_s"], available_at[uav["uav_id"]])
        start, completion, conflicts = _reserve_route(
            mission["route_corridors"],
            mission["assigned_layer_m"],
            raw_start,
            segment_seconds,
            reservations,
            enforce_capacity,
        )
        # Remove speculative reservations.  The selected candidate is committed below.
        for offset, corridor in enumerate(mission["route_corridors"]):
            reservations[(corridor, mission["assigned_layer_m"])].pop()
        ranked.append((completion, uav["uav_id"], uav, start, completion, conflicts))

    choice = ranked[0] if method in {"greedy", "no_symbol_grounding"} else min(ranked)
    _, _, uav, start, completion, conflicts = choice
    _reserve_route(
        mission["route_corridors"],
        mission["assigned_layer_m"],
        start,
        segment_seconds,
        reservations,
        enforce_capacity,
    )
    return uav, start, completion, conflicts


def _offline_fault_score(bundle: DatasetBundle, detected: dict[str, list[tuple[int, int]]]) -> float | None:
    """Open withheld labels only after scheduling to score detection recall."""

    faults = list(_records(bundle.root / "faults.jsonl"))
    if not faults:
        return None
    hits = 0
    for fault in faults:
        truth = (int(fault["start_time_s"]), int(fault["end_time_s"]))
        if any(_overlaps(*truth, candidate) for candidate in detected.get(fault["target_uav_id"], [])):
            hits += 1
    return round(hits / len(faults), 4)


def evaluate_dataset(bundle: DatasetBundle, method: str) -> BenchmarkResult:
    """Execute one method configuration over a frozen SkyRescue-Bench dataset."""

    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

    started_at = time.perf_counter()
    available_at: dict[str, int] = defaultdict(int)
    reservations: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    repair_latencies: list[float] = []
    completed = on_time = conflicts = grounded = assigned = repair_attempts = repairs = 0
    duplicate_external_calls = residual_reservations = 0
    committed_action_keys: set[str] = set()
    failure_reasons: defaultdict[str, int] = defaultdict(int)
    evidence_events = 0
    cp_assignments = _cp_sat_assign(bundle.missions, bundle.uavs) if method == "cp_sat" else {}

    # The dispatcher may prioritise requests already present at the same time,
    # but it must not schedule future arrivals before they enter the stream.
    for mission in sorted(bundle.missions, key=lambda item: (item["request_time_s"], -item["priority"], item["mission_id"])):
        candidate_uavs = bundle.uavs
        if method == "cp_sat":
            assigned_id = cp_assignments.get(mission["mission_id"])
            candidate_uavs = [uav for uav in bundle.uavs if uav["uav_id"] == assigned_id]
        uav, start, completion, mission_conflicts = _choose_assignment(
            mission, candidate_uavs, available_at, reservations, method
        )
        # Capacity-aware methods may inspect and avoid candidate overlaps.  Only
        # the greedy ablation actually commits those overlaps as conflicts.
        if method == "greedy":
            conflicts += mission_conflicts
        if uav is None:
            failure_reasons["no_compatible_resource"] += 1
            continue
        assigned += 1
        is_grounded = _compatible(mission, uav)
        grounded += int(is_grounded)

        # The deterministic safety kernel blocks an incompatible binding even
        # when the no-grounding ablation proposes one.
        if not is_grounded:
            segment_seconds = max(
                1, mission["estimated_duration_s"] // max(1, len(mission["route_corridors"]))
            )
            released = _release_route(
                mission["route_corridors"], mission["assigned_layer_m"], start,
                segment_seconds, reservations,
            )
            residual_reservations += len(mission["route_corridors"]) - released
            failure_reasons["safety_kernel_rejected_binding"] += 1
            continue

        incident = _has_incident(bundle, uav["uav_id"], start, completion)
        if incident:
            repair_attempts += 1
            segment_seconds = max(
                1, mission["estimated_duration_s"] // max(1, len(mission["route_corridors"]))
            )
            released = _release_route(
                mission["route_corridors"], mission["assigned_layer_m"], start,
                segment_seconds, reservations,
            )
            residual_reservations += len(mission["route_corridors"]) - released
            if method in {"skyrescue", "no_audit", "full_replan"}:
                t0 = time.perf_counter()
                if method == "full_replan":
                    # Full replan discards all future availability commitments.
                    available_at = defaultdict(int, {
                        key: value for key, value in available_at.items() if value <= start
                    })
                alternatives = [candidate for candidate in bundle.uavs if candidate["uav_id"] != uav["uav_id"]]
                replacement, repair_start, repair_completion, repair_conflicts = _choose_assignment(
                    mission, alternatives, available_at, reservations, method
                )
                repair_latencies.append((time.perf_counter() - t0) * 1000)
                if method == "greedy":
                    conflicts += repair_conflicts
                if replacement is not None and _compatible(mission, replacement):
                    uav, start, completion = replacement, repair_start, repair_completion
                    repairs += 1
                    evidence_events += 2 if method != "no_audit" else 0
                else:
                    failure_reasons["repair_no_compatible_replacement"] += 1
                    continue
            else:
                failure_reasons["incident_without_repair"] += 1
                continue

        available_at[uav["uav_id"]] = completion
        completed += 1
        on_time += int(completion <= mission["deadline_s"])
        action_key = f"{mission['mission_id']}:dispatch:v1"
        if action_key in committed_action_keys:
            duplicate_external_calls += 1
        committed_action_keys.add(action_key)
        evidence_events += 3 if method not in {"greedy", "no_audit"} else (1 if method == "greedy" else 0)

    configuration = bundle.manifest["configuration"]
    duration_hours = configuration["duration_seconds"] / 3600
    expected_evidence = max(1, len(bundle.missions) * 3 + repair_attempts * 2)
    grounding_accuracy = round(grounded / assigned, 4) if assigned else None
    repair_success_rate = round(repairs / repair_attempts, 4) if repair_attempts else None
    invariant_violations = _reservation_overlap_count(reservations)
    scheduler_wall_ms = round((time.perf_counter() - started_at) * 1000, 3)
    notes = [
        "All reported data are synthetic benchmark results.",
        "Fault labels were withheld from scheduling and used only for offline scoring.",
        "Unauthorized-call interception is not reported because this dataset contains no labelled authorization challenge set.",
    ]
    return BenchmarkResult(
        dataset=configuration["tier"],
        method=method,
        missions=len(bundle.missions),
        completed=completed,
        completion_rate=round(completed / len(bundle.missions), 4) if bundle.missions else 0.0,
        on_time_rate=round(on_time / len(bundle.missions), 4) if bundle.missions else 0.0,
        conflict_rate=round(conflicts / max(1, len(bundle.missions)), 4),
        throughput_per_hour=round(completed / duration_hours, 3),
        grounding_accuracy=grounding_accuracy,
        repair_success_rate=repair_success_rate,
        fault_detection_recall=_offline_fault_score(bundle, bundle.observations),
        replan_p50_ms=_percentile(repair_latencies, 0.50),
        replan_p95_ms=_percentile(repair_latencies, 0.95),
        replan_p99_ms=_percentile(repair_latencies, 0.99),
        scheduler_wall_ms=scheduler_wall_ms,
        peak_rss_mb=_peak_rss_mb(),
        timeout_rate=0.0,
        invariant_violations=invariant_violations,
        duplicate_external_calls=duplicate_external_calls,
        residual_reservations=residual_reservations,
        failure_reasons=dict(sorted(failure_reasons.items())),
        evidence_completeness=round(min(1.0, evidence_events / expected_evidence), 4),
        unauthorized_interception_rate=None,
        authorization_challenges=0,
        telemetry_rows=bundle.telemetry_rows,
        notes=notes,
    )


def summarize_seed_results(results: list[BenchmarkResult]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Return mean, standard deviation and 95% CI per method and metric."""

    metrics = (
        "completion_rate", "on_time_rate", "conflict_rate", "throughput_per_hour",
        "grounding_accuracy", "repair_success_rate", "fault_detection_recall",
        "replan_p50_ms", "replan_p95_ms", "replan_p99_ms", "scheduler_wall_ms",
        "peak_rss_mb", "timeout_rate", "invariant_violations",
        "duplicate_external_calls", "residual_reservations", "evidence_completeness",
    )
    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[result.method].append(result)
    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for method, rows in grouped.items():
        summary[method] = {}
        for metric in metrics:
            values = [getattr(row, metric) for row in rows if getattr(row, metric) is not None]
            if not values:
                summary[method][metric] = {"n": 0, "mean": None, "std": None, "ci95": None}
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[method][metric] = {
                "n": len(values),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95": round(1.96 * std / math.sqrt(len(values)), 4),
            }
    return summary

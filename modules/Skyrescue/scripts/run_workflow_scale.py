#!/usr/bin/env python3
"""Measure compilation and local repair on one size-controlled task graph.

``--size N`` means exactly N typed tasks in a *single* workflow graph. It does
not mean N repetitions of the small workflows in IntentSynth. Candidate
generation and event-fixture construction happen outside the timed regions;
typed validation, graph/planned-state construction, causal-closure traversal,
repair, and invariant checks are measured where documented in the protocol.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from skyrescue.workflow import RUNTIME_EVENT_TYPES, TASK_TYPES, ZONE_ALIASES  # noqa: E402


PRIORITIES = frozenset({"Critical", "High", "Normal"})
MISSION_STATE_TYPE = "MissionState"


@dataclass(frozen=True)
class TypedTask:
    """One statically typed task in the synthetic scale candidate."""

    task_id: str
    task_type: str
    target_zone: str
    priority: str
    skill: str
    input_type: str
    output_type: str
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class TypedCandidate:
    """A generated candidate; generation itself is deliberately not timed."""

    candidate_id: str
    declared_task_count: int
    tasks: tuple[TypedTask, ...]


@dataclass
class TaskRuntimeState:
    """Planned or committed binding state used by the scale mechanism."""

    phase: str
    binding: str
    reservation: str
    receipt: str | None


@dataclass
class WorkflowRuntimeState:
    version: int
    tasks: dict[str, TaskRuntimeState]


@dataclass(frozen=True)
class CompiledWorkflow:
    workflow_id: str
    tasks: dict[str, TypedTask]
    successors: dict[str, tuple[str, ...]]
    topological_order: tuple[str, ...]
    planned_state: WorkflowRuntimeState


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    event_type: str
    directly_affected: frozenset[str]


@dataclass(frozen=True)
class RepairObservation:
    impact_closure: tuple[str, ...]
    changed_nodes: int
    total_nodes: int
    change_ratio: float
    protected_commitments: int
    preserved_commitments: int
    commitment_preservation_rate: float


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("At least one observation is required")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("At least one observation is required")
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def generate_typed_candidate(size: int, seed: int) -> TypedCandidate:
    """Generate one connected, fixed-seed typed graph with exactly ``size`` tasks.

    The topology is a fork/join DAG: the first task releases independent
    mission tasks and the last task joins their evidence. Consequently, a
    middle-task event normally has a two-node causal closure, but that closure
    is discovered by traversing graph edges rather than by a size constant.
    """

    if size < 1:
        raise ValueError("Task graph size must be positive")
    task_types = sorted(TASK_TYPES)
    zones = sorted(ZONE_ALIASES)
    rng = random.Random(seed)
    tasks: list[TypedTask] = []
    task_ids = [f"task-{index:05d}" for index in range(size)]
    for index, task_id in enumerate(task_ids):
        task_type = task_types[rng.randrange(len(task_types))]
        if index == 0:
            dependencies: tuple[str, ...] = ()
        elif index == size - 1 and size > 2:
            dependencies = tuple(task_ids[1:-1])
        else:
            dependencies = (task_ids[0],)
        tasks.append(
            TypedTask(
                task_id=task_id,
                task_type=task_type,
                target_zone=zones[rng.randrange(len(zones))],
                priority="Critical",
                skill=TASK_TYPES[task_type]["skill"],
                input_type=MISSION_STATE_TYPE,
                output_type=MISSION_STATE_TYPE,
                dependencies=dependencies,
            )
        )
    return TypedCandidate(
        candidate_id=f"scale-{size}-{seed}",
        declared_task_count=size,
        tasks=tuple(tasks),
    )


def _topological_order(tasks: dict[str, TypedTask], successors: dict[str, list[str]]) -> tuple[str, ...]:
    indegree = {task_id: len(task.dependencies) for task_id, task in tasks.items()}
    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        task_id = heapq.heappop(ready)
        ordered.append(task_id)
        for dependent in successors[task_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(ordered) != len(tasks):
        raise ValueError("InvalidGraph: dependency cycle")
    return tuple(ordered)


def compile_typed_candidate(candidate: TypedCandidate) -> CompiledWorkflow:
    """Validate every task and materialize a graph plus side-effect-free plan.

    Compilation may select candidate bindings and create planned reservation
    records, but it must not claim an external effect was committed or mint an
    execution receipt.
    """

    if not isinstance(candidate, TypedCandidate):
        raise ValueError("InvalidSchema: candidate must be TypedCandidate")
    if candidate.declared_task_count != len(candidate.tasks) or not candidate.tasks:
        raise ValueError("InvalidSchema: declared task count does not match tasks")

    tasks: dict[str, TypedTask] = {}
    for task in candidate.tasks:
        if not isinstance(task, TypedTask) or not all(
            isinstance(value, str) and value
            for value in (
                task.task_id,
                task.task_type,
                task.target_zone,
                task.priority,
                task.skill,
                task.input_type,
                task.output_type,
            )
        ):
            raise ValueError("InvalidSchema: malformed typed task")
        if task.task_id in tasks:
            raise ValueError(f"InvalidGraph: duplicate task id {task.task_id}")
        if task.task_type not in TASK_TYPES:
            raise ValueError(f"InvalidType: unknown task type {task.task_type}")
        if task.input_type != MISSION_STATE_TYPE or task.output_type != MISSION_STATE_TYPE:
            raise ValueError(f"InvalidType: incompatible schema on {task.task_id}")
        if task.target_zone not in ZONE_ALIASES:
            raise ValueError(f"UngroundedEntity: {task.target_zone}")
        if task.priority not in PRIORITIES:
            raise ValueError(f"InvalidType: unknown priority {task.priority}")
        if task.skill != TASK_TYPES[task.task_type]["skill"]:
            raise ValueError(f"UnknownSkill: {task.skill} for {task.task_type}")
        tasks[task.task_id] = task

    successors: dict[str, list[str]] = {task_id: [] for task_id in tasks}
    edge_count = 0
    for task in candidate.tasks:
        if len(set(task.dependencies)) != len(task.dependencies):
            raise ValueError(f"InvalidGraph: duplicate dependency on {task.task_id}")
        for dependency in task.dependencies:
            if dependency == task.task_id or dependency not in tasks:
                raise ValueError(f"InvalidGraph: unresolved dependency {dependency}")
            if tasks[dependency].output_type != task.input_type:
                raise ValueError(f"InvalidType: edge {dependency}->{task.task_id}")
            successors[dependency].append(task.task_id)
            edge_count += 1

    order = _topological_order(tasks, successors)
    if len(tasks) > 1:
        if edge_count < len(tasks) - 1:
            raise ValueError("InvalidGraph: workflow is not connected")
        seen = {order[0]}
        frontier = [order[0]]
        while frontier:
            task_id = frontier.pop()
            neighbours = (*tasks[task_id].dependencies, *successors[task_id])
            for neighbour in neighbours:
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
        if len(seen) != len(tasks):
            raise ValueError("InvalidGraph: workflow is not connected")

    planned_state = WorkflowRuntimeState(
        version=0,
        tasks={
            task_id: TaskRuntimeState(
                phase="Planned",
                binding=f"resource:{task_id}:primary",
                reservation=f"planned-reservation:{candidate.candidate_id}:{task_id}",
                receipt=None,
            )
            for task_id in order
        },
    )
    return CompiledWorkflow(
        workflow_id=candidate.candidate_id,
        tasks=tasks,
        successors={task_id: tuple(sorted(values)) for task_id, values in successors.items()},
        topological_order=order,
        planned_state=planned_state,
    )


def build_committed_runtime_fixture(workflow: CompiledWorkflow) -> WorkflowRuntimeState:
    """Create an already-executed state for repair experiments.

    This synthetic fixture is deliberately separate from compilation and is
    built outside event timing. Its receipts represent effects that occurred
    before the injected event; they are not compiler output.
    """

    if set(workflow.planned_state.tasks) != set(workflow.tasks):
        raise ValueError("Invalid compiled plan: task set mismatch")
    committed: dict[str, TaskRuntimeState] = {}
    for task_id in workflow.topological_order:
        planned = workflow.planned_state.tasks[task_id]
        if planned.phase != "Planned" or planned.receipt is not None:
            raise ValueError("Invalid compiled plan: compilation contains execution evidence")
        committed[task_id] = TaskRuntimeState(
            phase="Committed",
            binding=planned.binding,
            reservation=f"reservation:{workflow.workflow_id}:{task_id}:v1",
            receipt=f"fixture-receipt:{workflow.workflow_id}:{task_id}:v1",
        )
    return WorkflowRuntimeState(version=1, tasks=committed)


def causal_impact_closure(
    workflow: CompiledWorkflow,
    directly_affected: frozenset[str],
) -> frozenset[str]:
    """Return the transitive successor closure of the affected task set."""

    if not directly_affected or not directly_affected <= workflow.tasks.keys():
        raise ValueError("Runtime event references an unknown or empty task set")
    closure = set(directly_affected)
    frontier = list(sorted(directly_affected))
    while frontier:
        source = frontier.pop()
        for dependent in workflow.successors[source]:
            if dependent not in closure:
                closure.add(dependent)
                frontier.append(dependent)
    return frozenset(closure)


def _task_snapshot(state: WorkflowRuntimeState, task_id: str) -> tuple[str, str, str, str | None]:
    task = state.tasks[task_id]
    return task.phase, task.binding, task.reservation, task.receipt


def compare_repair_states(
    before: WorkflowRuntimeState,
    after: WorkflowRuntimeState,
    closure: frozenset[str],
) -> RepairObservation:
    """Derive change and preservation metrics from actual before/after state."""

    all_tasks = set(before.tasks) | set(after.tasks)
    changed = sum(
        task_id not in before.tasks
        or task_id not in after.tasks
        or _task_snapshot(before, task_id) != _task_snapshot(after, task_id)
        for task_id in all_tasks
    )
    protected = [task_id for task_id in before.tasks if task_id not in closure]
    preserved = sum(
        task_id in after.tasks and _task_snapshot(before, task_id) == _task_snapshot(after, task_id)
        for task_id in protected
    )
    return RepairObservation(
        impact_closure=tuple(sorted(closure)),
        changed_nodes=changed,
        total_nodes=len(all_tasks),
        change_ratio=changed / len(all_tasks) if all_tasks else 0.0,
        protected_commitments=len(protected),
        preserved_commitments=preserved,
        commitment_preservation_rate=preserved / len(protected) if protected else 1.0,
    )


def recheck_invariants(
    workflow: CompiledWorkflow,
    before: WorkflowRuntimeState,
    after: WorkflowRuntimeState,
    closure: frozenset[str],
    touched: frozenset[str],
) -> None:
    """Check graph, resource, receipt, and closure-external commitment safety."""

    if set(before.tasks) != set(workflow.tasks) or set(after.tasks) != set(workflow.tasks):
        raise ValueError("I1 violation: workflow task set changed")
    if touched != closure:
        raise ValueError("I4 violation: repaired task set does not equal the causal closure")

    active_bindings: set[str] = set()
    for task_id in workflow.topological_order:
        original = before.tasks[task_id]
        current = after.tasks[task_id]
        if not current.binding or not current.reservation or not current.receipt:
            raise ValueError(f"I1 violation: incomplete post-event state for {task_id}")
        if current.binding in active_bindings:
            raise ValueError(f"I2 violation: duplicate active binding {current.binding}")
        active_bindings.add(current.binding)
        if task_id in closure:
            if current.phase != "Recovered":
                raise ValueError(f"I1 violation: repaired task did not reach Recovered: {task_id}")
            continue
        if original.phase != "Committed" or current.phase != "Committed":
            raise ValueError(f"I1 violation: closure-external task is not Committed: {task_id}")
        if _task_snapshot(before, task_id) != _task_snapshot(after, task_id):
            raise ValueError(f"I4 violation: closure-external commitment changed: {task_id}")


def execute_local_repair(
    workflow: CompiledWorkflow,
    before: WorkflowRuntimeState,
    after: WorkflowRuntimeState,
    event: RuntimeEvent,
    before_recheck: Callable[[WorkflowRuntimeState, frozenset[str]], None] | None = None,
) -> RepairObservation:
    """Compensate/rebind exactly the computed closure and observe its state delta.

    ``before_recheck`` exists solely as a fault-injection seam for invariant
    tests. Production benchmark calls never provide it.
    """

    if event.event_type not in RUNTIME_EVENT_TYPES:
        raise ValueError(f"Unknown runtime event: {event.event_type}")
    if before.version != after.version:
        raise ValueError("Event state versions do not match")
    closure = causal_impact_closure(workflow, event.directly_affected)
    touched: set[str] = set()
    for task_id in workflow.topological_order:
        if task_id not in closure:
            continue
        current = after.tasks[task_id]
        current.phase = "Compensating"
        current.reservation = f"released:{event.event_id}:{task_id}:v{before.version}"
        touched.add(task_id)

    for task_id in workflow.topological_order:
        if task_id not in closure:
            continue
        current = after.tasks[task_id]
        next_version = before.version + 1
        current.binding = f"resource:{task_id}:recovery:v{next_version}"
        current.reservation = f"reservation:{workflow.workflow_id}:{task_id}:v{next_version}"
        current.receipt = f"receipt:{event.event_id}:{task_id}:v{next_version}"
        current.phase = "Recovered"

    after.version = before.version + 1
    if before_recheck is not None:
        before_recheck(after, closure)
    recheck_invariants(workflow, before, after, closure, frozenset(touched))
    observation = compare_repair_states(before, after, closure)
    if observation.changed_nodes != len(closure):
        raise ValueError("I4 violation: observed changes do not equal the executed closure")
    return observation


def _event_for_repeat(workflow: CompiledWorkflow, seed: int, repeat: int) -> RuntimeEvent:
    rng = random.Random((seed << 16) ^ repeat ^ 0x5A17)
    task_count = len(workflow.topological_order)
    if task_count > 2:
        affected_index = rng.randrange(1, task_count - 1)
    else:
        affected_index = 0
    task_id = workflow.topological_order[affected_index]
    event_type = RUNTIME_EVENT_TYPES[repeat % len(RUNTIME_EVENT_TYPES)]
    return RuntimeEvent(
        event_id=f"event:{seed}:{repeat}:{event_type}",
        event_type=event_type,
        directly_affected=frozenset({task_id}),
    )


def run_one(
    intent_dataset: Path,
    size: int,
    seed: int,
    warmup_rounds: int = 5,
    repeats: int = 30,
) -> dict[str, Any]:
    """Run one size/seed cell; the parent places every cell in a fresh child."""

    if warmup_rounds < 0 or repeats < 1:
        raise ValueError("warmup_rounds must be nonnegative and repeats must be positive")
    dataset_file = intent_dataset / "intent_cases.jsonl"
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Frozen intent dataset is missing: {dataset_file}")
    candidate = generate_typed_candidate(size, seed)  # intentionally outside timing
    compile_values: list[float] = []
    event_values: list[float] = []
    closure_sizes: list[float] = []
    change_ratios: list[float] = []
    preservation_rates: list[float] = []
    sample_rows: list[dict[str, Any]] = []  # O(repeats), independent of graph size
    total_started = time.perf_counter_ns()

    workflow: CompiledWorkflow | None = None
    for repeat in range(warmup_rounds + repeats):
        started = time.perf_counter_ns()
        compiled = compile_typed_candidate(candidate)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if len(compiled.tasks) != size:
            raise ValueError("Compiler did not preserve the requested graph size")
        workflow = compiled
        if repeat >= warmup_rounds:
            compile_values.append(elapsed_ms)
    if workflow is None:  # guarded by repeats >= 1, retained for type checkers
        raise RuntimeError("No workflow was compiled")

    measured_repeats = range(-warmup_rounds, repeats)
    for repeat in measured_repeats:
        event_index = repeat if repeat >= 0 else repeats + repeat + 1_000_000
        event = _event_for_repeat(workflow, seed, event_index)
        before = build_committed_runtime_fixture(workflow)
        after = deepcopy(before)  # benchmark setup, excluded from event latency
        started = time.perf_counter_ns()
        observed = execute_local_repair(workflow, before, after, event)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if repeat < 0:
            continue
        event_values.append(elapsed_ms)
        closure_sizes.append(float(len(observed.impact_closure)))
        change_ratios.append(observed.change_ratio)
        preservation_rates.append(observed.commitment_preservation_rate)
        sample_rows.append(
            {
                "task_graph_size": size,
                "seed": seed,
                "repeat": repeat,
                "event_type": event.event_type,
                "directly_affected_count": len(event.directly_affected),
                "compile_ms": compile_values[repeat],
                "event_processing_ms": elapsed_ms,
                "repair_ms": elapsed_ms,
                "closure_size": len(observed.impact_closure),
                "changed_nodes": observed.changed_nodes,
                "total_nodes": observed.total_nodes,
                "change_ratio": observed.change_ratio,
                "protected_commitments": observed.protected_commitments,
                "preserved_commitments": observed.preserved_commitments,
                "commitment_preservation_rate": observed.commitment_preservation_rate,
            }
        )

    wall_ms = (time.perf_counter_ns() - total_started) / 1_000_000.0
    return {
        "run": {
            "task_graph_size": size,
            "workflow_graphs": 1,
            "seed": seed,
            "warmup_rounds": warmup_rounds,
            "measured_repeats": repeats,
            "wall_ms": wall_ms,
            "throughput_events_per_second": repeats / max(wall_ms / 1000.0, 1e-12),
            "peak_rss_mb": peak_rss_mb(),
            "compile": describe(compile_values),
            "event_processing": describe(event_values),
            "repair": describe(event_values),
            "closure_size": describe(closure_sizes),
            "change_ratio": describe(change_ratios),
            "commitment_preservation_rate": describe(preservation_rates),
        },
        "events": sample_rows,
    }


def flatten_run(run: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in run.items() if not isinstance(value, dict)}
    for group in (
        "compile",
        "event_processing",
        "repair",
        "closure_size",
        "change_ratio",
        "commitment_preservation_rate",
    ):
        for statistic, value in run[group].items():
            row[f"{group}_{statistic}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skyrescue-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    sizes = [row["task_graph_size"] for row in summary]
    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(sizes, [row["compile_p95_ms"] for row in summary], marker="o", label="Typed compile P95")
    axis.plot(sizes, [row["repair_p95_ms"] for row in summary], marker="s", label="Event repair P95")
    axis.set_xlabel("Task graph size (tasks)")
    axis.set_ylabel("Latency (ms)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "workflow_scale_latency.pdf")
    fig.savefig(output_dir / "workflow_scale_latency.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.errorbar(
        sizes,
        [row["peak_rss_mean_mb"] for row in summary],
        yerr=[row["peak_rss_sample_std_mb"] for row in summary],
        marker="o",
        capsize=3,
    )
    axis.set_xlabel("Task graph size (tasks)")
    axis.set_ylabel("Peak resident memory (MiB)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "workflow_scale_memory.pdf")
    fig.savefig(output_dir / "workflow_scale_memory.png", dpi=200)
    plt.close(fig)


def parent(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for size in args.sizes:
        for seed in args.seeds:
            completed = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--intent-dataset",
                    str(args.intent_dataset),
                    "--size",
                    str(size),
                    "--seed",
                    str(seed),
                    "--warmup-rounds",
                    str(args.warmup_rounds),
                    "--repeats",
                    str(args.repeats),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            payload = json.loads(completed.stdout)
            runs.append(payload["run"])
            events.extend(payload["events"])

    summary: list[dict[str, Any]] = []
    for size in args.sizes:
        size_events = [row for row in events if row["task_graph_size"] == size]
        size_runs = [row for row in runs if row["task_graph_size"] == size]
        compile_stats = describe([row["compile_ms"] for row in size_events])
        repair_stats = describe([row["repair_ms"] for row in size_events])
        closure_stats = describe([float(row["closure_size"]) for row in size_events])
        change_stats = describe([row["change_ratio"] for row in size_events])
        rss = [row["peak_rss_mb"] for row in size_runs]
        protected = sum(int(row["protected_commitments"]) for row in size_events)
        preserved = sum(int(row["preserved_commitments"]) for row in size_events)
        summary.append(
            {
                "task_graph_size": size,
                "workflow_graphs_per_cell": 1,
                "seeds": len(size_runs),
                "warmup_rounds_per_seed": args.warmup_rounds,
                "event_samples": len(size_events),
                "compile_p50_ms": compile_stats["p50"],
                "compile_p95_ms": compile_stats["p95"],
                "event_p50_ms": repair_stats["p50"],
                "event_p95_ms": repair_stats["p95"],
                "event_p99_ms": repair_stats["p99"],
                "repair_p50_ms": repair_stats["p50"],
                "repair_p95_ms": repair_stats["p95"],
                "repair_p99_ms": repair_stats["p99"],
                "closure_size_mean": closure_stats["mean"],
                "closure_size_p95": closure_stats["p95"],
                "change_ratio_mean": change_stats["mean"],
                "change_ratio_p95": change_stats["p95"],
                "commitment_preservation_rate": preserved / protected if protected else 1.0,
                "protected_commitments": protected,
                "preserved_commitments": preserved,
                "peak_rss_mean_mb": statistics.mean(rss),
                "peak_rss_sample_std_mb": statistics.stdev(rss) if len(rss) > 1 else 0.0,
            }
        )

    write_csv(args.output_dir / "workflow_scale_runs.csv", [flatten_run(row) for row in runs])
    write_csv(args.output_dir / "workflow_scale_events.csv", events)
    write_csv(args.output_dir / "workflow_scale_summary.csv", summary)
    payload = {
        "protocol": {
            "controlled_factor": "task count in one typed workflow task graph",
            "sizes": args.sizes,
            "seeds": args.seeds,
            "workflow_graphs_per_size_seed_cell": 1,
            "candidate_generation": "fixed-seed synthesis from the runtime type/skill/zone registries; excluded from compile timing",
            "compile_timing": "per-task schema/type/grounding/skill validation plus dependency DAG, candidate bindings, and planned reservations; no committed effects or receipts",
            "runtime_fixture": "already-executed committed state with synthetic pre-event receipts; constructed after compilation and outside event timing",
            "event_timing": "causal-closure traversal, closure-only compensation/rebinding to Recovered, invariant recheck, and observed before/after metrics",
            "warmup_rounds_per_seed": args.warmup_rounds,
            "measured_repeats_per_seed": args.repeats,
            "result_retention": "constant measured-repeat summaries per cell; no per-task rows retained",
            "memory": "each task-graph-size/seed cell executed in a fresh child process",
            "scope": "single-process synthetic mechanism scaling on one host; no distributed or flight-system claim",
        },
        "summary": summary,
    }
    (args.output_dir / "workflow_scale.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plots(args.output_dir, summary)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1000, 2000, 5000])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260811, 20260812, 20260813, 20260814, 20260815],
    )
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.size is None or args.seed is None:
            parser.error("--size and --seed are required with --worker")
        print(
            json.dumps(
                run_one(
                    args.intent_dataset,
                    args.size,
                    args.seed,
                    warmup_rounds=args.warmup_rounds,
                    repeats=args.repeats,
                ),
                ensure_ascii=False,
            )
        )
    elif args.output_dir:
        parent(args)
    else:
        parser.error("--output-dir is required unless --worker is set")


if __name__ == "__main__":
    main()

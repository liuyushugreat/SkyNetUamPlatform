#!/usr/bin/env python3
"""Measure mechanism and configured-stack latency for the SkyRescue prototype.

The configured-stack experiment is a matched 2x2 design: Native/LangGraph x
persistence off/on. All four cells receive the same payload, exclude framework
construction from the timed region, and, when persistence is on, write the same
payload immediately before and after one stack invocation. The resulting
differences are configured-stack costs, not intrinsic framework overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypedDict

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from skyrescue.runtime_latency import (  # noqa: E402
    candidate_from_frozen_case,
    compile_typed_candidate,
    full_replan,
    local_repair,
)
from skyrescue.security import evaluate_action  # noqa: E402
from skyrescue.workflow import build_runtime_event_profiles  # noqa: E402


MECHANISM_SPECS = (
    ("mechanism", "Typed intent compilation", "Native", "off", 300),
    ("mechanism", "Adjudication decision", "Native", "off", 600),
    ("mechanism", "SkyRescue local repair", "Native", "off", 164),
    ("mechanism", "Full-Replan", "Native", "off", 164),
    ("mechanism", "LangGraph-Workflow", "LangGraph", "on", 164),
)

CONFIGURED_STACKS = (
    ("Native", "off"),
    ("Native", "on"),
    ("LangGraph", "off"),
    ("LangGraph", "on"),
)

CONFIGURED_STACK_SPECS = tuple(
    ("configured_stack", "Configured-stack local repair", framework, persistence, 164)
    for framework, persistence in CONFIGURED_STACKS
)

RAW_FIELDS = (
    "benchmark_family",
    "mechanism",
    "framework",
    "persistence",
    "case_id",
    "repeat",
    "latency_ms",
)

SUMMARY_FIELDS = (
    "benchmark_family",
    "mechanism",
    "framework",
    "persistence",
    "n",
    "unique_samples",
    "repeats",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "sample_std_ms",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def timed(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot checkpoint value of type {type(value).__name__}")


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize the matched checkpoint payload deterministically."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


class MatchedSQLiteCheckpoint:
    """Two-boundary SQLite checkpoint used identically by both frameworks."""

    def __init__(self, database: Path):
        self.database = database
        self.connection = sqlite3.connect(database)
        self.connection.execute(
            """
            CREATE TABLE configured_stack_checkpoints (
              run_id TEXT NOT NULL,
              boundary TEXT NOT NULL CHECK(boundary IN ('input', 'output')),
              payload_json TEXT NOT NULL,
              PRIMARY KEY (run_id, boundary)
            )
            """
        )
        self.connection.commit()

    def save(self, run_id: str, boundary: str, payload: dict[str, Any]) -> None:
        if boundary not in {"input", "output"}:
            raise ValueError(f"Unknown checkpoint boundary: {boundary}")
        self.connection.execute(
            "INSERT INTO configured_stack_checkpoints(run_id, boundary, payload_json) VALUES(?, ?, ?)",
            (run_id, boundary, canonical_json(payload)),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class GraphState(TypedDict, total=False):
    case: dict[str, Any]
    event: dict[str, Any]
    recovered: dict[str, Any]


def build_langgraph(checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    def repair_node(state: GraphState) -> GraphState:
        return {"recovered": local_repair(state["case"], state["event"])}

    graph = StateGraph(GraphState)
    graph.add_node("repair", repair_node)
    graph.add_edge(START, "repair")
    graph.add_edge("repair", END)
    return graph.compile(checkpointer=checkpointer)


def invoke_configured_stack(
    framework: str,
    persistence: str,
    payload: GraphState,
    run_id: str,
    *,
    app=None,
    checkpoint: MatchedSQLiteCheckpoint | None = None,
) -> GraphState:
    """Invoke one matched configured-stack cell over a common payload boundary."""

    if framework not in {"Native", "LangGraph"}:
        raise ValueError(f"Unknown framework: {framework}")
    if persistence not in {"off", "on"}:
        raise ValueError(f"Unknown persistence setting: {persistence}")
    if (persistence == "on") != (checkpoint is not None):
        raise ValueError("Persistence-on requires exactly one matched SQLite checkpoint")

    if checkpoint is not None:
        checkpoint.save(run_id, "input", payload)

    if framework == "Native":
        result: GraphState = {
            "case": payload["case"],
            "event": payload["event"],
            "recovered": local_repair(payload["case"], payload["event"]),
        }
    else:
        if app is None:
            raise ValueError("LangGraph configured-stack cells require a compiled graph")
        result = app.invoke(
            payload,
            {"configurable": {"thread_id": run_id}},
        )

    if result.get("recovered", {}).get("status") != "Recovered":
        raise ValueError(f"Configured stack failed to recover {payload['case']['case_id']}")
    if checkpoint is not None:
        checkpoint.save(run_id, "output", result)
    return result


def summarize_rows(
    raw_rows: list[dict[str, Any]],
    specs: tuple[tuple[str, str, str, str, int], ...],
    repeats: int,
) -> list[dict[str, Any]]:
    """Summarize raw observations without pooling different stack settings."""

    summaries = []
    for family, mechanism, framework, persistence, unique_samples in specs:
        values = [
            float(row["latency_ms"])
            for row in raw_rows
            if row["benchmark_family"] == family
            and row["mechanism"] == mechanism
            and row["framework"] == framework
            and row["persistence"] == persistence
        ]
        expected = unique_samples * repeats
        if len(values) != expected:
            raise ValueError(
                f"Expected {expected} observations for {family}/{framework}/{persistence}, got {len(values)}"
            )
        summaries.append({
            "benchmark_family": family,
            "mechanism": mechanism,
            "framework": framework,
            "persistence": persistence,
            "n": len(values),
            "unique_samples": unique_samples,
            "repeats": repeats,
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99),
            "mean_ms": statistics.mean(values),
            "sample_std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-dataset", type=Path, required=True)
    parser.add_argument("--security-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    return parser


def _remove_sqlite_files(database: Path) -> None:
    for candidate in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
        if candidate.exists():
            candidate.unlink()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.warmup_rounds < 5 or args.repeats < 30:
        parser.error("at least 5 warm-up rounds and 30 measured repeats are required")

    cases = load_jsonl(args.intent_dataset / "intent_cases.jsonl")
    candidates = [candidate_from_frozen_case(case) for case in cases]
    requests = load_jsonl(args.security_dataset / "requests.jsonl")
    valid = [case for case in cases if case.get("expected_failure") is None]
    profiles = build_runtime_event_profiles(len(valid))
    recoverable = [
        (case, profile["event"])
        for case, profile in zip(valid, profiles)
        if profile["oracle"]["expected_outcome"] == "repair"
    ]
    if len(cases) != 300 or len(requests) != 600 or len(valid) != 172 or len(recoverable) != 164:
        raise ValueError("Frozen dataset cardinality does not match the paper protocol")

    configured_payloads: list[GraphState] = [
        {"case": case, "event": event}
        for case, event in recoverable
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []

    def record_row(
        family: str,
        mechanism: str,
        framework: str,
        persistence: str,
        case_id: str,
        repeat: int,
        latency: float,
    ) -> None:
        raw_rows.append({
            "benchmark_family": family,
            "mechanism": mechanism,
            "framework": framework,
            "persistence": persistence,
            "case_id": case_id,
            "repeat": repeat,
            "latency_ms": latency,
        })

    def run_compiler(record: bool, repeat: int) -> None:
        for case, candidate in zip(cases, candidates):
            result, latency = timed(lambda candidate=candidate: compile_typed_candidate(candidate))
            if result["kind"] not in {"ExecutableWorkflow", "StructuredFailure"}:
                raise ValueError("Compiler returned neither workflow nor structured failure")
            if record:
                record_row("mechanism", "Typed intent compilation", "Native", "off", case["case_id"], repeat, latency)

    def run_security(record: bool, repeat: int) -> None:
        seen: set[str] = set()
        for request in requests:
            (_, _), latency = timed(lambda request=request: evaluate_action(request, seen))
            if record:
                record_row("mechanism", "Adjudication decision", "Native", "off", request["request_id"], repeat, latency)

    def run_runtime(function, mechanism: str, record: bool, repeat: int) -> None:
        for case, event in recoverable:
            result, latency = timed(lambda case=case, event=event: function(case, event))
            if result.get("status") != "Recovered":
                raise ValueError(f"{mechanism} failed to recover {case['case_id']}")
            if record:
                record_row("mechanism", mechanism, "Native", "off", case["case_id"], repeat, latency)

    for warmup in range(1, args.warmup_rounds + 1):
        run_compiler(False, warmup)
        run_security(False, warmup)
        run_runtime(local_repair, "SkyRescue local repair", False, warmup)
        run_runtime(full_replan, "Full-Replan", False, warmup)

    for repeat in range(1, args.repeats + 1):
        run_compiler(True, repeat)
        run_security(True, repeat)
        run_runtime(local_repair, "SkyRescue local repair", True, repeat)
        run_runtime(full_replan, "Full-Replan", True, repeat)

    # Preserve the original persisted LangGraph mechanism measurement.
    from langgraph.checkpoint.sqlite import SqliteSaver

    for phase, rounds in (("warmup", args.warmup_rounds), ("measure", args.repeats)):
        for repeat in range(1, rounds + 1):
            database = args.output_dir / f"langgraph_{phase}_{repeat}.sqlite"
            _remove_sqlite_files(database)
            with sqlite3.connect(database, check_same_thread=False) as connection:
                app = build_langgraph(SqliteSaver(connection))
                for payload in configured_payloads:
                    case = payload["case"]
                    result, latency = timed(
                        lambda payload=payload, case=case: app.invoke(
                            payload,
                            {"configurable": {"thread_id": f"legacy-{phase}-{repeat}-{case['case_id']}"}},
                        )
                    )
                    if result.get("recovered", {}).get("status") != "Recovered":
                        raise ValueError(f"LangGraph-Workflow failed to recover {case['case_id']}")
                    if phase == "measure":
                        record_row("mechanism", "LangGraph-Workflow", "LangGraph", "on", case["case_id"], repeat, latency)
            _remove_sqlite_files(database)

    # Matched 2x2 configured-stack experiment. Both persistence-on cells use
    # exactly two explicit SQLite checkpoints around the same invocation payload.
    for phase, rounds in (("warmup", args.warmup_rounds), ("measure", args.repeats)):
        for repeat in range(1, rounds + 1):
            # Rotate cell order deterministically to limit fixed-order thermal bias.
            offset = (repeat - 1) % len(CONFIGURED_STACKS)
            ordered_stacks = CONFIGURED_STACKS[offset:] + CONFIGURED_STACKS[:offset]
            for framework, persistence in ordered_stacks:
                app = build_langgraph(None) if framework == "LangGraph" else None
                database = args.output_dir / f"configured_stack_{phase}_{repeat}_{framework.lower()}_{persistence}.sqlite"
                checkpoint = None
                if persistence == "on":
                    _remove_sqlite_files(database)
                    checkpoint = MatchedSQLiteCheckpoint(database)
                try:
                    for payload in configured_payloads:
                        case_id = payload["case"]["case_id"]
                        run_id = f"{phase}-{repeat}-{case_id}"
                        (_, latency) = timed(
                            lambda framework=framework, persistence=persistence, payload=payload, run_id=run_id, app=app, checkpoint=checkpoint: invoke_configured_stack(
                                framework,
                                persistence,
                                payload,
                                run_id,
                                app=app,
                                checkpoint=checkpoint,
                            )
                        )
                        if phase == "measure":
                            record_row(
                                "configured_stack",
                                "Configured-stack local repair",
                                framework,
                                persistence,
                                case_id,
                                repeat,
                                latency,
                            )
                finally:
                    if checkpoint is not None:
                        checkpoint.close()
                        _remove_sqlite_files(database)

    raw_path = args.output_dir / "SkyRescue_runtime_benchmark.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({**row, "latency_ms": f"{row['latency_ms']:.6f}"})

    summary_rows = summarize_rows(
        raw_rows,
        MECHANISM_SPECS + CONFIGURED_STACK_SPECS,
        args.repeats,
    )
    summary_path = args.output_dir / "SkyRescue_runtime_benchmark_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                **row,
                "p50_ms": f"{row['p50_ms']:.6f}",
                "p95_ms": f"{row['p95_ms']:.6f}",
                "p99_ms": f"{row['p99_ms']:.6f}",
                "mean_ms": f"{row['mean_ms']:.6f}",
                "sample_std_ms": f"{row['sample_std_ms']:.6f}",
            })

    metadata = {
        "experiment": "Configured-stack 2x2 latency under matched payload and checkpoint boundaries",
        "interpretation": (
            "Measured differences reflect the configured orchestration and persistence stacks "
            "and must not be interpreted as intrinsic framework overhead."
        ),
        "frameworks": ["Native", "LangGraph"],
        "persistence": ["off", "on"],
        "payload_boundary": "Identical case/observable-event input and recovered-state output in all four cells",
        "checkpoint_boundary": "Persistence-on writes immediately before and after one stack invocation",
        "warmup_rounds": args.warmup_rounds,
        "repeats": args.repeats,
    }
    (args.output_dir / "SkyRescue_runtime_benchmark_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metadata": metadata, "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

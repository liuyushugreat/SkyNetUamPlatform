#!/usr/bin/env python3
"""Measure the five process-local mechanisms reported in SkyRescue Table 8."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypedDict

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from skyrescue.runtime_latency import candidate_from_frozen_case, compile_typed_candidate, full_replan, local_repair
from skyrescue.security import evaluate_action
from skyrescue.workflow import build_runtime_event_profiles, compile_case


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def timed(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


class GraphState(TypedDict, total=False):
    case: dict[str, Any]
    profile: dict[str, Any]
    recovered: dict[str, Any]


def build_langgraph(checkpointer):
    from langgraph.graph import END, START, StateGraph

    def repair_node(state: GraphState) -> GraphState:
        return {"recovered": local_repair(state["case"], state["profile"])}

    graph = StateGraph(GraphState)
    graph.add_node("repair", repair_node)
    graph.add_edge(START, "repair")
    graph.add_edge("repair", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-dataset", type=Path, required=True)
    parser.add_argument("--security-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmup_rounds < 5 or args.repeats < 10:
        parser.error("at least 5 warm-up rounds and 10 measured repeats are required")

    cases = load_jsonl(args.intent_dataset / "intent_cases.jsonl")
    candidates = [candidate_from_frozen_case(case) for case in cases]
    requests = load_jsonl(args.security_dataset / "requests.jsonl")
    valid = [case for case in cases if case.get("expected_failure") is None]
    profiles = build_runtime_event_profiles(len(valid))
    recoverable = [(case, profile) for case, profile in zip(valid, profiles) if profile["recoverable"]]
    if len(cases) != 300 or len(requests) != 600 or len(valid) != 172 or len(recoverable) != 164:
        raise ValueError("Frozen dataset cardinality does not match the paper protocol")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []

    def run_compiler(record: bool, repeat: int) -> None:
        for case, candidate in zip(cases, candidates):
            result, latency = timed(lambda candidate=candidate: compile_typed_candidate(candidate))
            if result["kind"] not in {"ExecutableWorkflow", "StructuredFailure"}:
                raise ValueError("Compiler returned neither workflow nor structured failure")
            if record:
                raw_rows.append({"mechanism": "Typed intent compilation", "case_id": case["case_id"], "repeat": repeat, "latency_ms": latency})

    def run_security(record: bool, repeat: int) -> None:
        seen: set[str] = set()
        for request in requests:
            (_, _), latency = timed(lambda request=request: evaluate_action(request, seen))
            if record:
                raw_rows.append({"mechanism": "Proposal-Adjudication-Commit", "case_id": request["request_id"], "repeat": repeat, "latency_ms": latency})

    def run_runtime(function, mechanism: str, record: bool, repeat: int) -> None:
        for case, profile in recoverable:
            result, latency = timed(lambda case=case, profile=profile: function(case, profile))
            if result.get("status") != "Recovered":
                raise ValueError(f"{mechanism} failed to recover {case['case_id']}")
            if record:
                raw_rows.append({"mechanism": mechanism, "case_id": case["case_id"], "repeat": repeat, "latency_ms": latency})

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

    from langgraph.checkpoint.sqlite import SqliteSaver

    for phase, rounds in (("warmup", args.warmup_rounds), ("measure", args.repeats)):
        for repeat in range(1, rounds + 1):
            database = args.output_dir / f"langgraph_{phase}_{repeat}.sqlite"
            if database.exists():
                database.unlink()
            with sqlite3.connect(database, check_same_thread=False) as connection:
                app = build_langgraph(SqliteSaver(connection))
                for case, profile in recoverable:
                    result, latency = timed(
                        lambda case=case, profile=profile: app.invoke(
                            {"case": case, "profile": profile},
                            {"configurable": {"thread_id": f"{phase}-{repeat}-{case['case_id']}"}},
                        )
                    )
                    if result.get("recovered", {}).get("status") != "Recovered":
                        raise ValueError(f"LangGraph failed to recover {case['case_id']}")
                    if phase == "measure":
                        raw_rows.append({"mechanism": "LangGraph-Workflow", "case_id": case["case_id"], "repeat": repeat, "latency_ms": latency})
            database.unlink()

    raw_path = args.output_dir / "SkyRescue_runtime_benchmark.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mechanism", "case_id", "repeat", "latency_ms"])
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({**row, "latency_ms": f"{row['latency_ms']:.6f}"})

    order = [
        ("Typed intent compilation", 300),
        ("Proposal-Adjudication-Commit", 600),
        ("SkyRescue local repair", 164),
        ("Full-Replan", 164),
        ("LangGraph-Workflow", 164),
    ]
    summary_rows = []
    for mechanism, unique_samples in order:
        values = [float(row["latency_ms"]) for row in raw_rows if row["mechanism"] == mechanism]
        summary_rows.append({
            "mechanism": mechanism,
            "n": len(values),
            "unique_samples": unique_samples,
            "repeats": args.repeats,
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
        })
    summary_path = args.output_dir / "SkyRescue_runtime_benchmark_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mechanism", "n", "unique_samples", "repeats", "p50_ms", "p95_ms"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({**row, "p50_ms": f"{row['p50_ms']:.6f}", "p95_ms": f"{row['p95_ms']:.6f}"})
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inject real child-process crashes into the SQLite durable-runtime prototype."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.durable_runtime import CRASH_EXIT_CODE, DurableWorkflowRuntime


def worker(database: Path, workflow_id: str, crash: bool) -> None:
    runtime = DurableWorkflowRuntime(database)
    try:
        runtime.start(workflow_id)
        runtime.execute(workflow_id, crash_after_effect=crash)
    finally:
        if not crash:
            runtime.close()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))]


def parent(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for trial in range(1, args.trials + 1):
        database = args.output_dir / f"trial_{trial:03d}.sqlite"
        workflow_id = f"crash-trial-{trial:03d}"
        crash = subprocess.run(
            [sys.executable, __file__, "--worker", "--database", str(database), "--workflow-id", workflow_id, "--crash"],
            check=False,
        )
        started = time.perf_counter()
        resume = subprocess.run(
            [sys.executable, __file__, "--worker", "--database", str(database), "--workflow-id", workflow_id],
            check=False,
        )
        recovery_ms = (time.perf_counter() - started) * 1000
        runtime = DurableWorkflowRuntime(database)
        try:
            state = runtime.inspect(workflow_id)
        finally:
            runtime.close()
        records.append({
            "trial": trial,
            "crash_exit_code": crash.returncode,
            "resume_exit_code": resume.returncode,
            "recovery_ms": round(recovery_ms, 4),
            **state,
        })

    completed = [row for row in records if row["workflow_status"] == "Committed" and row["operation_state"] == "Committed"]
    payload = {
        "experiment": "SQLite simulated external sink; real child-process termination after effect and before local receipt",
        "trials": args.trials,
        "crash_exit_code_expected": CRASH_EXIT_CODE,
        "recovery_success_rate": round(len(completed) / args.trials, 4),
        "duplicate_external_calls": sum(max(0, row["effect_count"] - 1) for row in records),
        "state_restoration_accuracy": round(sum(row["workflow_version"] == 2 for row in records) / args.trials, 4),
        "reservation_consistency": round(sum(row["reservation_consistent"] for row in records) / args.trials, 4),
        "evidence_chain_continuity": round(sum(row["evidence_chain_continuous"] for row in records) / args.trials, 4),
        "recovery_p50_ms": round(percentile([row["recovery_ms"] for row in records], 0.50), 4),
        "recovery_p95_ms": round(percentile([row["recovery_ms"] for row in records], 0.95), 4),
        "recovery_p99_ms": round(percentile([row["recovery_ms"] for row in records], 0.99), 4),
        "records": records,
        "boundary": "This is a single-host SQLite prototype with a simulated idempotent receiver. It does not establish distributed crash tolerance, real-UAV deployment, or flight safety.",
    }
    (args.output_dir / "crash_recovery_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--workflow-id")
    parser.add_argument("--crash", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.database, args.workflow_id, args.crash)
    elif args.output_dir:
        parent(args)
    else:
        parser.error("--output-dir is required unless --worker is set")


if __name__ == "__main__":
    main()

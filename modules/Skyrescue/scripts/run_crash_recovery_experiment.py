#!/usr/bin/env python3
"""Inject three real process-crash windows into the SQLite runtime contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.durable_runtime import CRASH_EXIT_CODE, CrashPoint, DurableWorkflowRuntime


def worker(database: Path, workflow_id: str, crash_point: str | None) -> None:
    runtime = DurableWorkflowRuntime(database)
    try:
        runtime.start(workflow_id)
        runtime.execute(workflow_id, crash_point=crash_point)
    finally:
        if crash_point is None:
            runtime.close()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))]


def parent(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    windows = list(CrashPoint)
    for window in windows:
        for trial in range(1, args.trials + 1):
            database = args.output_dir / f"{window.value}_trial_{trial:03d}.sqlite"
            workflow_id = f"{window.value}-{trial:03d}"
            crashed = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--database",
                    str(database),
                    "--workflow-id",
                    workflow_id,
                    "--crash-point",
                    window.value,
                ],
                check=False,
            )
            if crashed.returncode != CRASH_EXIT_CODE:
                raise RuntimeError(
                    f"{window.value} trial {trial} returned {crashed.returncode}, expected {CRASH_EXIT_CODE}"
                )
            started = time.perf_counter()
            resumed = subprocess.run(
                [sys.executable, __file__, "--worker", "--database", str(database), "--workflow-id", workflow_id],
                check=False,
            )
            recovery_ms = (time.perf_counter() - started) * 1000
            if resumed.returncode != 0:
                raise RuntimeError(f"{window.value} trial {trial} did not resume: {resumed.returncode}")
            runtime = DurableWorkflowRuntime(database)
            try:
                state = runtime.inspect(workflow_id)
            finally:
                runtime.close()
            records.append(
                {
                    "window": window.value,
                    "trial": trial,
                    "crash_exit_code": crashed.returncode,
                    "resume_exit_code": resumed.returncode,
                    "recovery_ms": round(recovery_ms, 4),
                    **state,
                }
            )
            if not args.keep_databases:
                database.unlink()

    by_window = {}
    for window in windows:
        subset = [row for row in records if row["window"] == window.value]
        completed = [
            row
            for row in subset
            if row["workflow_status"] == "Committed" and row["operation_state"] == "Committed"
        ]
        by_window[window.value] = {
            "trials": len(subset),
            "recovery_success_rate": round(len(completed) / len(subset), 4),
            "duplicate_invocations": sum(max(0, row["invoke_count"] - 1) for row in subset),
            "duplicate_effects": sum(max(0, row["effect_count"] - 1) for row in subset),
            "receipt_count_violations": sum(row["receipt_count"] != 1 for row in subset),
            "recovery_p50_ms": round(percentile([row["recovery_ms"] for row in subset], 0.50), 4),
            "recovery_p95_ms": round(percentile([row["recovery_ms"] for row in subset], 0.95), 4),
            "recovery_p99_ms": round(percentile([row["recovery_ms"] for row in subset], 0.99), 4),
        }

    payload = {
        "experiment": "Three non-atomic commit windows with real child-process termination",
        "receiver": (
            "single-host SQLite simulation with receiver deduplication and an HMAC-authenticated "
            "receipt bound to the idempotency key, issue version, and causal parent"
        ),
        "trials_per_window": args.trials,
        "total_crashes": len(records),
        "crash_exit_code_expected": CRASH_EXIT_CODE,
        "windows": by_window,
        "state_restoration_accuracy": round(sum(row["workflow_version"] == 2 for row in records) / len(records), 4),
        "receipt_binding_validity": round(
            sum(row["receiver_receipt_valid"] and row["local_receipt_valid"] for row in records)
            / len(records),
            4,
        ),
        "reservation_consistency": round(sum(row["reservation_consistent"] for row in records) / len(records), 4),
        "evidence_chain_continuity": round(sum(row["evidence_chain_continuous"] for row in records) / len(records), 4),
        "records": records,
        "boundary": (
            "This is a single-host SQLite prototype with a simulated queryable, key-deduplicating receiver. "
            "It does not establish distributed crash tolerance, receiver-query availability, real-UAV deployment, or flight safety."
        ),
    }
    (args.output_dir / "crash_recovery_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trials", type=int, default=30, help="trials per crash window")
    parser.add_argument("--keep-databases", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--workflow-id")
    parser.add_argument("--crash-point", choices=[point.value for point in CrashPoint])
    parser.add_argument("--crash", action="store_true", help="legacy alias for after_effect_before_receipt")
    args = parser.parse_args()
    if args.worker:
        point = CrashPoint.AFTER_EFFECT_BEFORE_RECEIPT.value if args.crash else args.crash_point
        worker(args.database, args.workflow_id, point)
    elif args.output_dir:
        parent(args)
    else:
        parser.error("--output-dir is required unless --worker is set")


if __name__ == "__main__":
    main()

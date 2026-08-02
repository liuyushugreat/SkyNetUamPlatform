#!/usr/bin/env python3
"""Run a lightweight multi-seed workflow-runtime scale experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.workflow import summarize_scale


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def run_one(size: int, seed: int) -> dict:
    rng = random.Random(seed)
    arrivals = sorted(rng.randrange(max(60, size // 2)) for _ in range(size))
    queue = 0
    max_queue = 0
    transitions = 0
    events = 0
    evidence_hash = b"0" * 32
    started = time.perf_counter()
    cursor = 0
    for tick in range(max(arrivals) + 1):
        while cursor < size and arrivals[cursor] == tick:
            cursor += 1
            queue += 1
            events += 1
        capacity = 1 + rng.randrange(4)
        processed = min(queue, capacity)
        queue -= processed
        for workflow_offset in range(processed):
            for state_offset in range(12):
                evidence_hash = hashlib.sha256(
                    evidence_hash
                    + f"{seed}:{tick}:{cursor + workflow_offset}:{state_offset}".encode("ascii")
                ).digest()
                transitions += 1
        if rng.random() < 0.08 and processed:
            events += 1
            for state_offset in range(5):
                evidence_hash = hashlib.sha256(evidence_hash + bytes([state_offset])).digest()
                transitions += 1
        max_queue = max(max_queue, queue)
    for workflow_offset in range(queue):
        for state_offset in range(12):
            evidence_hash = hashlib.sha256(
                evidence_hash + f"tail:{workflow_offset}:{state_offset}".encode("ascii")
            ).digest()
            transitions += 1
    events += queue
    wall = (time.perf_counter() - started) * 1000
    seconds = max(wall / 1000, 1e-9)
    return {
        "workflows": size,
        "seed": seed,
        "events": events,
        "state_transitions": transitions,
        "wall_ms": round(wall, 4),
        "events_per_second": round(events / seconds, 2),
        "transitions_per_second": round(transitions / seconds, 2),
        "max_queue": max_queue,
        "peak_rss_mb": round(peak_rss_mb(), 4),
        "timeout_rate": 0.0,
        "final_evidence_hash": evidence_hash.hex(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000, 2000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260811, 20260812, 20260813, 20260814, 20260815])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_one(size, seed) for size in args.sizes for seed in args.seeds]
    summary = summarize_scale(rows)
    (args.output_dir / "workflow_scale.json").write_text(
        json.dumps({"runs": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "workflow_scale.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate intent compilation and workflow repair on frozen JSONL cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.workflow import evaluate_compilers, evaluate_runtime


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: dict[str, dict]):
    fields = sorted({key for metrics in rows.values() for key in metrics})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *fields])
        writer.writeheader()
        for method, metrics in rows.items():
            writer.writerow({"method": method, **metrics})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.dataset / "intent_cases.jsonl")
    compiler = evaluate_compilers(cases)
    runtime = evaluate_runtime(cases)
    payload = {
        "dataset": json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8")),
        "compiler_results": compiler,
        "runtime_results": runtime,
        "evidence_boundary": (
            "Model-free synthetic benchmark; direct_text is not a Direct-LLM baseline, "
            "and generator labels are not a human gold set."
        ),
    }
    (args.output_dir / "workflow_benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "compiler_results.csv", compiler)
    write_csv(args.output_dir / "runtime_results.csv", runtime)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

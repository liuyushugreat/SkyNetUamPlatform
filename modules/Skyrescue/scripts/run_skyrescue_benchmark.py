#!/usr/bin/env python3
"""Run SkyRescue ablations on frozen SkyRescue-Bench datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.benchmark import METHODS, evaluate_dataset, load_dataset, summarize_seed_results


def main() -> None:
    parser = argparse.ArgumentParser(description="SkyRescue-Bench evaluator")
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in args.datasets:
        bundle = load_dataset(path)
        print(
            f"Loaded {bundle.manifest['configuration']['tier']}: "
            f"{len(bundle.missions)} missions, {bundle.telemetry_rows} telemetry rows"
        )
        for method in args.methods:
            result = evaluate_dataset(bundle, method)
            results.append(result)
            print(
                f"  {method:20s} completion={result.completion_rate:.3f} "
                f"conflict={result.conflict_rate:.3f} repair={result.repair_success_rate}"
            )

    payload = [result.to_dict() for result in results]
    (args.output_dir / "all_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key, value in payload[0].items() if key != "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload:
            writer.writerow({key: value for key, value in row.items() if key != "notes"})

    seed_results = [result for result in results if result.dataset.startswith("large_seed_")]
    summary = summarize_seed_results(seed_results)
    (args.output_dir / "large_seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()

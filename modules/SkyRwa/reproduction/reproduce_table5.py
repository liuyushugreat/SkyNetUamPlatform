"""Reproduce Table 5: Benchmark dataset — 105 flights across 10 scenarios.

Paper Section 7.1: Benchmark Dataset
Generates the full benchmark, prints scenario distribution, and exports
JSON + RDF data to the data/ directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.benchmarks.generate_benchmark import generate, SCENARIOS


def main():
    print("=" * 72)
    print("  Table 5: Benchmark — 105 flights across 10 scenarios")
    print("=" * 72)

    output_dir = SCRIPT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = generate(output_dir)

    print(f"\nTotal flights: {summary['total_flights']}")
    print(f"Graph triples: {summary['graph_triples']}")
    print(f"Tradable:      {summary['tradable_count']}")
    print(f"Non-tradable:  {summary['non_tradable_count']}")
    print(f"With violations: {summary['with_violations']}")
    print()

    print(f"{'Scenario':<30} {'Flights':>8}")
    print("-" * 40)
    for tag in sorted(summary["scenario_counts"]):
        print(f"{tag:<30} {summary['scenario_counts'][tag]:>8}")
    print("-" * 40)
    print(f"{'TOTAL':<30} {summary['total_flights']:>8}")

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nData written to: {output_dir.relative_to(SCRIPT_DIR)}/")
    print("  benchmark_flights.json  — input flight records")
    print("  benchmark_assets.json   — processed asset units")
    print("  benchmark_labels.json   — expected labels")
    print("  benchmark_summary.json  — summary statistics")

    graph_dir = output_dir.parent / "data"
    print(f"\nRDF graphs in: {output_dir.relative_to(SCRIPT_DIR)}/")
    print("  benchmark_graph.ttl     — Turtle serialization")
    print("  benchmark_graph.jsonld  — JSON-LD serialization")

    return summary


if __name__ == "__main__":
    main()

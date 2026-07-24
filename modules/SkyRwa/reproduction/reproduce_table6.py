"""Reproduce Table 6: Baseline comparison — JSON-scan vs SPARQL.

Paper Section 7.2: Baseline
Compares JSON file scanning vs SPARQL on four query tasks:
  1. Find tradable assets
  2. Revenue by participant
  3. Governance violations
  4. Product lineage (3-hop traversal)

Reports lines-of-code (LoC) and execution time for each approach.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_baseline_comparison import run_comparison
from SkyRwa.benchmarks.generate_benchmark import generate


def main():
    print("=" * 78)
    print("  Table 6: Baseline Comparison — JSON-scan vs SPARQL")
    print("  (Paper Section 7.2)")
    print("=" * 78)
    print()

    data_dir = SCRIPT_DIR / "data"
    sample_dir = SCRIPT_DIR.parent / "benchmarks" / "sample_data"
    if not (sample_dir / "benchmark_assets.json").exists():
        print("[INFO] Generating benchmark data first...")
        generate(data_dir)

    run_comparison()


if __name__ == "__main__":
    main()

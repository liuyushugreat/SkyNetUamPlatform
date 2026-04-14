"""Reproduce Semantic Baseline: Lifecycle KG vs Flat KG.

Paper Section 7.3: Semantic Baseline
Compares a flat RDF graph (all entities as generic prov:Entity) with
the lifecycle KG (typed tiers) on four audit tasks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_semantic_baseline import run_semantic_baseline
from SkyRwa.benchmarks.generate_benchmark import generate


def main():
    print("=" * 78)
    print("  Semantic Baseline: Lifecycle KG vs Flat KG")
    print("  (Paper Section 7.3)")
    print("=" * 78)
    print()

    sample_dir = SCRIPT_DIR.parent / "benchmarks" / "sample_data"
    if not (sample_dir / "benchmark_assets.json").exists():
        print("[INFO] Generating benchmark data first...")
        generate()

    results = run_semantic_baseline()

    out_dir = SCRIPT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "semantic_baseline.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_dir / 'semantic_baseline.json'}")


if __name__ == "__main__":
    main()

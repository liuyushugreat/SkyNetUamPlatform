"""Reproduce Table 8: Scalability — overhead and graph size for 5–1000 flights.

Paper Section 7.4: Scalability
Measures:
  - Pipeline runtime (ms)
  - RDF mapping time (ms)
  - Turtle serialization time (ms)
  - SHACL validation time (ms)
  - Triple count and triples-per-flight ratio

Expected: linear growth for pipeline/RDF; superlinear for SHACL; ~66 triples/flight.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_overhead import run


def main():
    print("=" * 72)
    print("  Table 8: Scalability — Overhead for 5–1000 flights")
    print("  (Paper Section 7.4)")
    print("=" * 72)
    print()

    results = run()

    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table8_scalability.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to: outputs/table8_scalability.json")


if __name__ == "__main__":
    main()

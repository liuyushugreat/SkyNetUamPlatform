"""Reproduce Table 9: Competency questions — SPARQL queryability.

Paper Section 7.5: SPARQL Queryability
Runs 6 competency questions (CQ1–CQ6) and 4 analytical queries against
the benchmark knowledge graph.

Expected: All CQs return correct results; CQ3/CQ6 demonstrate multi-hop traversal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.run_queries import run


def main():
    print("=" * 72)
    print("  Table 9: SPARQL Competency Questions (CQ1–CQ6)")
    print("  (Paper Section 7.5)")
    print("=" * 72)
    print()

    results = run()

    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table9_queryability.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to: outputs/table9_queryability.json")


if __name__ == "__main__":
    main()

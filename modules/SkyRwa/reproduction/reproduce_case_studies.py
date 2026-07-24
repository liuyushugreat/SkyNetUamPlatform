"""Reproduce Section 7.6: Case studies.

Paper Section 7.6: Case Studies
Three detailed cases:
  1. Clean promotion — 12 flights pass governance, get aggregated into product
  2. Governance failure cascade — NFZ flights blocked from trading
  3. Audit task comparison — SPARQL vs JSON for cross-entity lineage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_case_studies import run


def main():
    print("=" * 72)
    print("  Section 7.6: Case Studies")
    print("=" * 72)
    print()

    results = run()

    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "case_studies.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to: outputs/case_studies.json")


if __name__ == "__main__":
    main()

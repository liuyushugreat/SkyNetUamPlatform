"""Reproduce SHACL & governance rule validation coverage.

Paper Section 5: Governance and Validation
Runs SHACL shapes and SPARQL governance rules against the full benchmark
graph, reporting violation counts and rule coverage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_validation import run


def main():
    print("=" * 72)
    print("  SHACL & Governance Rule Validation Coverage")
    print("  (Paper Section 5)")
    print("=" * 72)
    print()

    results = run()

    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_coverage.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to: outputs/validation_coverage.json")


if __name__ == "__main__":
    main()

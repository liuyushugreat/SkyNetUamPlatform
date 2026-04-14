"""Reproduce Ontology Quality Assessment.

Paper Section 4.5: Ontology Quality Assessment
Runs OOPS!-style pitfall scanning, reasoner consistency checks,
and CQ → ontology construct mapping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_ontology_quality import run_ontology_quality


def main():
    print("=" * 78)
    print("  Ontology Quality Assessment")
    print("  (Paper Section 4.5)")
    print("=" * 78)
    print()

    results = run_ontology_quality()

    out_dir = SCRIPT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ontology_quality.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

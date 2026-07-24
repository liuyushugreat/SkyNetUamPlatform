"""Reproduce robustness and sensitivity analysis.

Paper Section 7: Evaluation — Robustness
Three experiments:
  1. Multi-run stability (5 seeds, deterministic outcomes)
  2. Scale sensitivity (10–1000 flights, triples/flight ratio)
  3. Governance threshold sensitivity (5x5 grid)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_robustness import run


def main():
    print("=" * 72)
    print("  Robustness & Sensitivity Analysis")
    print("  (Paper Section 7 — Supplementary)")
    print("=" * 72)
    print()

    results = run()

    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to: outputs/robustness.json")


if __name__ == "__main__":
    main()

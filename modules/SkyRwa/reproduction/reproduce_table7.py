"""Reproduce paper Tables 8-9: validation-layer coverage — Python vs SHACL vs Combined.

Paper Section 7.3: Governance Ablation Study
Tests 6 violation types under three configurations:
  - Python-only rules
  - SHACL-only shapes
  - Combined dual-layer approach

Demonstrates that neither layer alone catches all violations.
Expected result: Combined detects 83%+ of violation types.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_ablation import run_ablation


def main():
    print("=" * 80)
    print("  Table 7: Governance Ablation — Python vs SHACL vs Combined")
    print("  (Paper Section 7.3)")
    print("=" * 80)
    print()

    run_ablation()


if __name__ == "__main__":
    main()

"""Reproduce Pilot Expert Evaluation results.

Paper Section 7.8: Pilot Expert Evaluation
Reports task completion time, correctness, confidence, and perceived
auditability for 4 domain experts × 4 tasks × 3 interfaces.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_user_study import run_user_study


def main():
    print("=" * 78)
    print("  Pilot Expert Evaluation")
    print("  (Paper Section 7.8)")
    print("=" * 78)
    print()

    run_user_study()


if __name__ == "__main__":
    main()

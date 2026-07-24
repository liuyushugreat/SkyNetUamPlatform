"""Reproduce Section 7.8: end-to-end walkthrough of one blocked flight
(FLT-NFZ-DEMO, ingest to GOV-001 blocking).

Regenerates outputs/walkthrough.json and outputs/walkthrough_generated.tex
(the latter is \\input by the paper, keeping paper and code in sync).
Runtime: ~5 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.evaluation.walkthrough import run

if __name__ == "__main__":
    run()

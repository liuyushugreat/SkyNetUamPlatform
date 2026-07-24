"""Reproduce Table 11 (and the appendix mapping, Table 12): verification of
all twelve competency questions against the enriched 105-flight audit graph.

Paper Section 7.7.  Builds the enriched graph (products, usage events,
settlements, decisions; ~9k triples), runs CQ1-CQ12, and compares each
result set against ground truth computed from the domain objects.
Writes outputs/competency.json.  Runtime: ~10 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_competency import run

if __name__ == "__main__":
    run()

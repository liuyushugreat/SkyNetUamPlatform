"""Reproduce Table 9 (and the SHACL+ctx column of Table 8): measured cost
of scoring-context materialization.

Paper Section 7.4: extended declarative contract (V3/V4/V6 as
SHACL-SPARQL) vs. the baseline contract, at 5-1000 flights.
Writes outputs/scoring_context.json.  Runtime: ~6 min (the slowest step).
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_scoring_context import run

if __name__ == "__main__":
    run()

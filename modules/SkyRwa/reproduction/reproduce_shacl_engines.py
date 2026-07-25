"""Reproduce the rudof columns of Table 11 and Figure 2: dual-engine SHACL
comparison (pySHACL vs rudof) at 5-1000 flights.

Paper Section 7.5.  Requires ``pip install pyrudof``.
Writes outputs/shacl_engines.json, then renders figs/shacl_engines.pdf.
Runtime: ~1.5 min.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_pkg_root = SCRIPT_DIR.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.experiments.eval_shacl_engines import run
from SkyRwa.experiments.plot_shacl_engines import main as plot

if __name__ == "__main__":
    run()
    plot()

"""Plot the SHACL engine comparison (pySHACL vs rudof) as a log-log figure.

Reads ISWC2026/outputs/shacl_engines.json (produced by
eval_shacl_engines.py) and writes shacl_engines.pdf next to it.

Fonts are embedded as TrueType (pdf.fonttype=42) to avoid Type 3 fonts,
which many publisher submission systems reject.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt

OUTPUTS = Path(__file__).resolve().parent.parent / "ISWC2026" / "outputs"
DATA = OUTPUTS / "shacl_engines.json"
FIG = OUTPUTS / "shacl_engines.pdf"


def main() -> None:
    results = json.loads(DATA.read_text(encoding="utf-8"))
    sizes = results["sizes"]
    runs = results["runs"]

    pyshacl = [runs[f"n{n}"]["pyshacl_ms"] for n in sizes]
    rudof_total = [runs[f"n{n}"]["rudof_total_ms"] for n in sizes]
    rudof_val = [runs[f"n{n}"]["rudof_validate_ms"] for n in sizes]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.loglog(sizes, pyshacl, "o-", color="#1f77b4",
              label="pySHACL (Core + SPARQL)")
    ax.loglog(sizes, rudof_total, "s-", color="#d62728",
              label="rudof, parse + validate (Core)")
    ax.loglog(sizes, rudof_val, "s--", color="#d62728", alpha=0.55,
              label="rudof, validate only (Core)")

    ax.set_xlabel("Number of flights")
    ax.set_ylabel("Validation time (ms)")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG)
    print(f"Saved: {FIG}")


if __name__ == "__main__":
    main()

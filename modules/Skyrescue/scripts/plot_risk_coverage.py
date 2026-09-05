#!/usr/bin/env python3
"""Render the frozen-accept-only HeldOut100 risk--coverage sensitivity plot."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Risk--coverage CSV is empty")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skyrescue-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.4, 3.4))
    labels = {"deepseek": "DeepSeek", "qwen": "Qwen"}
    markers = {"deepseek": "o", "qwen": "s"}
    for provider in sorted({row["provider"] for row in rows}):
        subset = [row for row in rows if row["provider"] == provider]
        coverage = [100 * float(row["coverage"]) for row in subset]
        risk = [100 * float(row["selective_risk"]) for row in subset]
        axis.plot(coverage, risk, marker=markers.get(provider, "o"), label=labels.get(provider, provider))
        axis.annotate("frozen", (coverage[0], risk[0]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Coverage (%)")
    axis.set_ylabel("Selective risk (%)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_xlim(18.5, 23.5)
    fig.tight_layout()
    fig.savefig(args.output_dir / "heldout_risk_coverage.pdf")
    fig.savefig(args.output_dir / "heldout_risk_coverage.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

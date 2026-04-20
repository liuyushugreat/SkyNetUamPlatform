"""Render the figures used by the SkyCert paper from ``outputs/metrics.json``.

Outputs:
    outputs/figs/coverage_vs_threat.pdf    — bar chart of coverage/ECE.
    outputs/figs/martingale_max.pdf        — log-scale martingale peak.
    outputs/figs/critical_error.pdf        — critical-error-rate before/after abstain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skycert.config import SkyCertConfig
from skycert.utils import ensure_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render SkyCert figures")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to write rendered figures (defaults to <config.output_dir>/figs)",
    )
    args = parser.parse_args(argv)

    config = SkyCertConfig.load(args.config)
    out_dir = Path(config.output_dir)
    with open(out_dir / "metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    runs = metrics["runs"]
    figs = ensure_dir(
        Path(args.output_dir) if args.output_dir is not None else out_dir / "figs"
    )

    names = [r["threat"]["name"] for r in runs]
    coverage = [r["coverage"] for r in runs]
    ece = [r["ece"] for r in runs]
    mart = [r["martingale_max"] for r in runs]
    crit_before = [r["critical_error_rate_base"] for r in runs]
    crit_after = [r["critical_error_rate_after_abstain"] for r in runs]
    abstain = [r["abstain_rate"] for r in runs]

    # Figure 1: coverage and ECE per threat.
    fig, ax1 = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width / 2, coverage, width, label="Coverage",
            color="#4C72B0")
    ax1.axhline(1 - config.assurance.conformal.alpha, color="black",
                linestyle="--", linewidth=0.8, label="Target 1-α")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Empirical coverage")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, ece, width, label="ECE", color="#DD8452")
    ax2.set_ylabel("Expected Calibration Error")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "coverage_vs_threat.pdf")
    plt.close(fig)

    # Figure 2: martingale peak (log-scale).
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(x, mart, color="#55A868")
    ax.set_yscale("log")
    ax.axhline(config.assurance.martingale.threshold, color="red",
               linestyle="--", linewidth=0.8, label="Alert threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Max martingale value (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "martingale_max.pdf")
    plt.close(fig)

    # Figure 3: critical error rate before/after abstention + abstain rate.
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(x - width / 2, crit_before, width, label="Critical error (no abstain)",
           color="#C44E52")
    ax.bar(x + width / 2, crit_after, width, label="Critical error (after abstain)",
           color="#8172B2")
    for xi, a in zip(x, abstain):
        ax.text(xi, max(crit_before[0], 0.01) + 0.02,
                f"abstain={a:.2f}", ha="center", fontsize=8, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Critical-class miss rate")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "critical_error.pdf")
    plt.close(fig)

    print(f"[SkyCert] wrote figures under {figs}")


if __name__ == "__main__":
    main()

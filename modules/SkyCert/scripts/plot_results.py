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

    # Figure 4: Pareto curve (abstain rate vs critical error) if baselines.json exists.
    baseline_path = out_dir / "baselines.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as fh:
            bl = json.load(fh)
        pareto = bl["pareto"]
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        for key, label, color, marker in [
            ("msp", "MSP threshold", "#C44E52", "v"),
            ("entropy", "Entropy threshold", "#DD8452", "^"),
            ("skycert", "SkyCert (sweep γ)", "#4C72B0", "o"),
        ]:
            pts = pareto.get(key, [])
            if not pts:
                continue
            xs = [p["abstain_rate"] for p in pts]
            ys = [p["critical_error_after_abstain"] for p in pts]
            ax.plot(xs, ys, marker=marker, markersize=4, label=label,
                    color=color, linewidth=1.2, alpha=0.85)
        bl_items = bl["baselines"]
        for item in bl_items:
            if item["method"] == "full SkyCert":
                ax.plot(item["abstain_rate"],
                        item["critical_error_rate_after_abstain"],
                        marker="*", markersize=12, color="#4C72B0",
                        zorder=5)
        ax.set_xlabel("Abstention rate")
        ax.set_ylabel("Critical-class miss rate (after abstention)")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(0.0, None)
        fig.tight_layout()
        fig.savefig(figs / "pareto_abstention.pdf")
        plt.close(fig)

    # Figure 5: lambda_drift sweep and beta4 attack-strength sweep, if
    # extensions.json is present.
    ext_path = out_dir / "extensions.json"
    if ext_path.exists():
        with open(ext_path, "r", encoding="utf-8") as fh:
            ext = json.load(fh)

        # Lambda sweep.
        lam_runs = ext.get("lambda_sweep", [])
        if lam_runs:
            fig, ax = plt.subplots(figsize=(5.0, 3.0))
            lams = [r["lambda_drift"] for r in lam_runs]
            crit = [r["critical_error_rate_after_abstain"] for r in lam_runs]
            abst = [r["abstain_rate"] for r in lam_runs]
            ax.plot(lams, crit, marker="o", color="#4C72B0",
                    label="Critical miss (after abstention)")
            ax.plot(lams, abst, marker="s", color="#8172B2",
                    label="Abstention rate", alpha=0.8)
            ax.set_xlabel(r"$\lambda_{\mathrm{drift}}$")
            ax.set_ylabel("Rate")
            ax.legend(fontsize=8, loc="best")
            ax.set_ylim(0.0, None)
            fig.tight_layout()
            fig.savefig(figs / "lambda_sweep.pdf")
            plt.close(fig)

        # Attack-strength sweeps.
        sweep = ext.get("attack_strength_sweep", {})
        if sweep:
            fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
            for ax, key, xlabel in [
                (axes[0], "beta3", r"$\beta_3$ (feature attack $\ell_\infty$)"),
                (axes[1], "beta4", r"$\beta_4$ (covariate shift strength)"),
            ]:
                runs = sweep.get(key, [])
                xs = [r["strength"] for r in runs]
                base = [r["critical_error_rate_base"] for r in runs]
                after = [r["critical_error_rate_after_abstain"] for r in runs]
                abst = [r["abstain_rate"] for r in runs]
                ax.plot(xs, base, marker="o", color="#C44E52",
                        label="Critical miss (no abstain)")
                ax.plot(xs, after, marker="s", color="#8172B2",
                        label="Critical miss (after abstain)")
                ax.plot(xs, abst, marker="^", color="#4C72B0",
                        linestyle=":", label="Abstain rate", alpha=0.8)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Rate")
                ax.set_ylim(0.0, None)
                ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(figs / "attack_strength_sweep.pdf")
            plt.close(fig)

    print(f"[SkyCert] wrote figures under {figs}")


if __name__ == "__main__":
    main()

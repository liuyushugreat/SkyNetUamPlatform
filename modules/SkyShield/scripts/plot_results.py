"""Render the four Matplotlib figures and the stage-latency heatmap.

* fig_timing_cdf.pdf        - stage + end-to-end latency CDFs (E2).
* fig_stress_tradeoff.pdf   - deadline-miss vs. tail-latency per regime (E3).
* fig_multi_radar_scaling.pdf - E4 scaling surface.
* fig_ablation_bars.pdf     - mission success / deadline miss per variant.
* fig_safety_ci.pdf         - E6 Clopper-Pearson confidence intervals.

All figures are plain PDF — no raster art, no conference-specific
branding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------------------------------------------------------------- #
# Figures                                                                    #
# -------------------------------------------------------------------------- #


def plot_timing(out_dir: Path, figs_dir: Path) -> None:
    data = _load(out_dir / "timing.json")
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for s, xs in data["samples"].items():
        xs = sorted(xs)
        if not xs:
            continue
        y = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, y, label=s.replace("_", " "))
    e2e = sorted(data["end_to_end_samples_ms"])
    if e2e:
        y = np.arange(1, len(e2e) + 1) / len(e2e)
        ax.plot(e2e, y, linestyle="--", linewidth=2, color="black",
                label="end-to-end")
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("SkyShield stage + end-to-end latency CDFs")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_timing_cdf.pdf")
    plt.close(fig)


def plot_stress(out_dir: Path, figs_dir: Path) -> None:
    data = _load(out_dir / "replay_stress.json")
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for r in data["regimes"]:
        ax.scatter(r["p99_ms"], r["deadline_miss"] * 100,
                   s=45, alpha=0.85)
        ax.annotate(r["regime"], (r["p99_ms"], r["deadline_miss"] * 100),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("P99 end-to-end latency (ms)")
    ax.set_ylabel("Deadline-miss rate (%)")
    ax.set_title("E3 replay stress regimes")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_stress_tradeoff.pdf")
    plt.close(fig)


def plot_multi_radar(out_dir: Path, figs_dir: Path) -> None:
    data = _load(out_dir / "multi_radar.json")
    if data is None:
        return
    rows = data["rows"]
    radars = sorted({r["num_radars"] for r in rows})
    concs = sorted({r["target_concurrency"] for r in rows})
    grid = np.zeros((len(concs), len(radars)))
    grid_miss = np.zeros_like(grid)
    for r in rows:
        i = concs.index(r["target_concurrency"])
        j = radars.index(r["num_radars"])
        grid[i, j] = r["p99_latency_ms_mean"]
        grid_miss[i, j] = r["deadline_miss_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    im0 = axes[0].imshow(grid, aspect="auto", cmap="viridis",
                         origin="lower")
    axes[0].set_xticks(range(len(radars)))
    axes[0].set_xticklabels(radars)
    axes[0].set_yticks(range(len(concs)))
    axes[0].set_yticklabels(concs)
    axes[0].set_xlabel("Radar count")
    axes[0].set_ylabel("Target concurrency")
    axes[0].set_title("P99 end-to-end latency (ms)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(grid_miss * 100, aspect="auto", cmap="magma",
                         origin="lower")
    axes[1].set_xticks(range(len(radars)))
    axes[1].set_xticklabels(radars)
    axes[1].set_yticks(range(len(concs)))
    axes[1].set_yticklabels(concs)
    axes[1].set_xlabel("Radar count")
    axes[1].set_ylabel("Target concurrency")
    axes[1].set_title("Deadline miss (%)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_multi_radar_scaling.pdf")
    plt.close(fig)


def plot_ablation(out_dir: Path, figs_dir: Path) -> None:
    data = _load(out_dir / "ablation.json")
    if data is None:
        return
    rows = data["variants"]
    names = [r["variant"] for r in rows]
    x = np.arange(len(names))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(6.4, 3.2))
    ax1.bar(x - width / 2, [r["mission_success"] for r in rows], width,
            label="mission success", color="#3a6ea5")
    ax1.bar(x + width / 2, [r["deadline_miss"] for r in rows], width,
            label="deadline miss", color="#c44d58")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("rate (0-1)")
    ax1.set_title("E5 ablation (mission success vs. deadline miss)")
    ax1.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_ablation_bars.pdf")
    plt.close(fig)


def plot_safety(out_dir: Path, figs_dir: Path) -> None:
    data = _load(out_dir / "safety.json")
    if data is None:
        return
    rows = data["rows"]
    names = [r["scenario"] for r in rows]
    rates = [r["rate"] for r in rows]
    lo = [r["ci95_lo"] for r in rows]
    hi = [r["ci95_hi"] for r in rows]
    yerr = np.vstack([np.array(rates) - np.array(lo),
                      np.array(hi) - np.array(rates)])
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.errorbar(names, rates, yerr=yerr, fmt="o", capsize=4, color="#225577")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Correct-response rate")
    ax.set_title("E6 safety scenarios (95% CI over 100 trials)")
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_safety_ci.pdf")
    plt.close(fig)


# -------------------------------------------------------------------------- #
# CLI                                                                         #
# -------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.outputs)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_timing(out_dir, figs_dir)
    plot_stress(out_dir, figs_dir)
    plot_multi_radar(out_dir, figs_dir)
    plot_ablation(out_dir, figs_dir)
    plot_safety(out_dir, figs_dir)

    print(f"[plot] wrote figures to {figs_dir}")


if __name__ == "__main__":
    main()

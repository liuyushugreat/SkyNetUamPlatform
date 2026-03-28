"""Generate paper figures: ablation CDR bar chart, latency comparison,
precision-recall scatter, scalability curves, and transfer CDR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


METHODS_ORDER = ["VO", "LSTM-P", "Tfm-P", "STGCN", "GAT-S", "TR-GAT-NT", "TR-GAT"]
METHOD_COLORS = {
    "VO": "#95a5a6", "LSTM-P": "#e67e22", "Tfm-P": "#f39c12",
    "STGCN": "#27ae60", "GAT-S": "#2980b9", "TR-GAT-NT": "#8e44ad",
    "TR-GAT": "#e74c3c",
}


def plot_ablation_cdr(
    results: Dict[str, Dict],
    output_path: str | Path = "charts/fig_ablation_cdr.png",
) -> None:
    """Bar chart of CDR across methods with error bars."""
    if not HAS_MPL:
        return
    _ensure_dir(output_path)

    methods = [m for m in METHODS_ORDER if m in results]
    cdrs = [results[m]["cdr_mean"] for m in methods]
    stds = [results[m].get("cdr_std", 0) for m in methods]
    colors = [METHOD_COLORS.get(m, "#333") for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, cdrs, yerr=stds, color=colors, capsize=4, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Conflict Detection Rate (CDR)", fontsize=12)
    ax.set_title("CDR Comparison Across Methods", fontsize=14)
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, cdrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def plot_latency_comparison(
    results: Dict[str, Dict],
    output_path: str | Path = "charts/fig_latency_comparison.png",
) -> None:
    """Horizontal bar chart of 95th-pctl latency."""
    if not HAS_MPL:
        return
    _ensure_dir(output_path)

    methods = [m for m in METHODS_ORDER if m in results]
    latencies = [results[m]["latency_mean"] for m in methods]
    colors = [METHOD_COLORS.get(m, "#333") for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(methods))
    ax.barh(y_pos, latencies, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("95th Percentile Latency (ms)", fontsize=12)
    ax.set_title("Inference Latency Comparison", fontsize=14)
    ax.axvline(x=200, color="red", linestyle="--", alpha=0.5, label="200ms budget")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    for i, val in enumerate(latencies):
        ax.text(val + 2, i, f"{val:.1f}ms", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def plot_precision_recall_scatter(
    results: Dict[str, Dict],
    output_path: str | Path = "charts/fig_precision_recall_scatter.png",
) -> None:
    """Precision vs Recall scatter plot."""
    if not HAS_MPL:
        return
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHODS_ORDER:
        if method not in results:
            continue
        r = results[method]
        ax.scatter(
            r.get("recall_mean", r.get("cdr_mean", 0)),
            r.get("precision_mean", 1 - r.get("far_mean", 0)),
            s=120, color=METHOD_COLORS.get(method, "#333"),
            label=method, edgecolors="black", linewidth=0.5, zorder=3,
        )

    ax.set_xlabel("Recall (CDR)", fontsize=12)
    ax.set_ylabel("Precision (1 - FAR)", fontsize=12)
    ax.set_title("Precision–Recall Operating Points", fontsize=14)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def plot_scalability(
    scalability_data: Dict[int, Dict[str, float]],
    output_path: str | Path = "charts/fig_scalability_latency.png",
) -> None:
    """Latency vs fleet size for TR-GAT and baselines."""
    if not HAS_MPL:
        return
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    fleet_sizes = sorted(scalability_data.keys())

    for method in ["VO", "STGCN", "GAT-S", "TR-GAT"]:
        latencies = []
        for fs in fleet_sizes:
            if method in scalability_data[fs]:
                latencies.append(scalability_data[fs][method])
            else:
                latencies.append(None)
        valid = [(fs, lat) for fs, lat in zip(fleet_sizes, latencies) if lat is not None]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid],
                    "o-", label=method, color=METHOD_COLORS.get(method, "#333"), markersize=6)

    ax.axhline(y=200, color="red", linestyle="--", alpha=0.5, label="200ms budget")
    ax.set_xlabel("Fleet Size (number of UAVs)", fontsize=12)
    ax.set_ylabel("95th Percentile Latency (ms)", fontsize=12)
    ax.set_title("Scalability: Latency vs Fleet Size", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def plot_main_results_table(
    results: Dict[str, Dict],
    output_path: str | Path = "charts/fig_main_results.png",
) -> None:
    """Render main results as a publication-quality table figure."""
    if not HAS_MPL:
        return
    _ensure_dir(output_path)

    methods = [m for m in METHODS_ORDER if m in results]
    headers = ["Method", "CDR ↑", "FAR ↓", "F1 ↑", "Latency (ms) ↓"]

    cell_text = []
    for m in methods:
        r = results[m]
        cell_text.append([
            m,
            f"{r['cdr_mean']:.4f} ± {r.get('cdr_std', 0):.4f}",
            f"{r['far_mean']:.4f} ± {r.get('far_std', 0):.4f}",
            f"{r['f1_mean']:.4f} ± {r.get('f1_std', 0):.4f}",
            f"{r['latency_mean']:.1f} ± {r.get('latency_std', 0):.1f}",
        ])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    best_row = len(methods)
    for j in range(len(headers)):
        table[(best_row, j)].set_facecolor("#fadbd8")

    plt.title("Table 1: Overall detection performance on UrbanAir-500 test set", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_figures(results: Dict[str, Dict], output_dir: str | Path = "charts"):
    """Generate all paper figures from aggregated results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ablation_cdr(results, output_dir / "fig_ablation_cdr.png")
    plot_latency_comparison(results, output_dir / "fig_latency_comparison.png")
    plot_precision_recall_scatter(results, output_dir / "fig_precision_recall_scatter.png")
    plot_main_results_table(results, output_dir / "fig_main_results.png")


def _ensure_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

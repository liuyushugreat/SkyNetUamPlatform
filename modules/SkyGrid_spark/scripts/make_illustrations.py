"""Render the two non-data illustrations (``fig_dag.pdf`` and
``fig_arch.pdf``) referenced by the IEEE HPCC 2026 paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from _common import MODULE_ROOT


def draw_dag(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    ops = [
        ("feat_extract", "NN", 1.0, 1.0, "#ffd38a"),
        ("risk_score",   "NN", 3.0, 1.5, "#ffd38a"),
        ("rule_check",   "sym", 5.2, 1.0, "#cce6b8"),
        ("conformal",    "sym", 7.2, 1.0, "#cce6b8"),
        ("audit",        "sym", 9.0, 1.0, "#cce6b8"),
    ]
    pos = {}
    for name, kind, x, y, c in ops:
        box = mpatches.FancyBboxPatch(
            (x - 0.85, y - 0.30), 1.7, 0.60,
            boxstyle="round,pad=0.02", linewidth=1.0,
            edgecolor="#444", facecolor=c,
        )
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=9)
        ax.text(x, y - 0.50, kind, ha="center", va="center",
                fontsize=7, color="#666")
        pos[name] = (x, y)

    def arrow(src, dst):
        (x0, y0), (x1, y1) = pos[src], pos[dst]
        ax.annotate(
            "", xy=(x1 - 0.85, y1), xytext=(x0 + 0.85, y0),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.1,
                            connectionstyle="arc3,rad=0.0"),
        )

    for src, dst in [
        ("feat_extract", "risk_score"),
        ("feat_extract", "rule_check"),
        ("risk_score", "rule_check"),
        ("rule_check", "conformal"),
        ("conformal", "audit"),
    ]:
        arrow(src, dst)

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def draw_arch(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Layers
    ax.text(0.4, 6.6, "SkyGridRuntime (engine.py)", fontsize=10, weight="bold")

    # Control plane (top row)
    for (x, y, w, h, label, color) in [
        (1.0, 5.4, 2.2, 0.7, "STP\npartitioner", "#e8d0f0"),
        (4.0, 5.4, 2.2, 0.7, "COP\nplacer",      "#dceaf5"),
        (7.0, 5.4, 2.2, 0.7, "ABP\npipeline",    "#d9f0d2"),
        (10.0, 5.4, 1.6, 0.7, "Tracer\n+Metrics", "#f5e3d0"),
    ]:
        box = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            linewidth=1.0, edgecolor="#444", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    # Fabric (bottom row)
    for (x, y, w, h, label, color) in [
        (1.0, 2.8, 2.2, 1.1, "Cloud\n(2.4 TFLOPS)", "#f8d3d2"),
        (4.0, 2.8, 1.4, 1.1, "Edge-0", "#d3dff6"),
        (5.8, 2.8, 1.4, 1.1, "Edge-1", "#d3dff6"),
        (7.6, 2.8, 1.4, 1.1, "Edge-2", "#d3dff6"),
        (9.4, 2.8, 1.4, 1.1, "Edge-3", "#d3dff6"),
    ]:
        box = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            linewidth=1.0, edgecolor="#444", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    ax.text(0.4, 1.4, "Discrete-Event Simulator (queues + links + jitter)",
            fontsize=9, weight="bold", color="#333")
    base = mpatches.FancyBboxPatch(
        (0.8, 0.4), 10.6, 0.7,
        boxstyle="round,pad=0.02", linewidth=1.0,
        edgecolor="#444", facecolor="#eeeeee",
    )
    ax.add_patch(base)
    ax.text(6.1, 0.75, "network: edge-cloud 12 ms / 1 Gbps   \u00B7   "
                       "edge-edge 4 ms / 0.5 Gbps   \u00B7   Gaussian jitter",
            ha="center", va="center", fontsize=8)

    # Arrows between layers
    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.0))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label,
                    ha="center", va="bottom", fontsize=7, color="#555")

    arrow(2.1, 5.4, 2.1, 3.9, "assignment $\\pi$")
    arrow(5.1, 5.4, 5.1, 3.9, "placement $\\sigma$")
    arrow(8.1, 5.4, 8.1, 3.9, "batches")

    # Event flow
    ax.annotate("", xy=(1.0, 3.3), xytext=(-0.5, 3.3),
                arrowprops=dict(arrowstyle="-|>", color="#a04040", lw=1.4))
    ax.text(-0.5, 3.5, "events", fontsize=8, color="#a04040")

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out = MODULE_ROOT / "outputs" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    draw_dag(out / "fig_dag.pdf")
    draw_arch(out / "fig_arch.pdf")
    print(f"[plot] wrote fig_dag.pdf, fig_arch.pdf to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

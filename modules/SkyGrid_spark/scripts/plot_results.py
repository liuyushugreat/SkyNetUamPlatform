"""Render the paper figures from ``metrics.json`` / ``ablation.json`` /
``scaling.json``.  Produces PDFs into ``outputs/figs/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import MODULE_ROOT

sys.path.insert(0, str(MODULE_ROOT))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig_throughput_latency(metrics: dict, out: Path) -> None:
    runs = metrics["runs"]
    labels = [r["spec"]["label"] for r in runs]
    tputs = [r["metrics"]["throughput_ops"] for r in runs]
    p99s  = [r["metrics"]["latency_ms"]["p99"] for r in runs]

    fig, axL = plt.subplots(figsize=(8.2, 3.4))
    x = np.arange(len(labels))
    w = 0.38
    axL.bar(x - w/2, tputs, width=w, color="#3b6aa0", label="throughput (ops/s)")
    axL.set_ylabel("throughput (ops/s)")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axR = axL.twinx()
    axR.bar(x + w/2, p99s, width=w, color="#c25b56", label="p99 latency (ms)")
    axR.set_ylabel("p99 latency (ms)")

    # Single combined legend.
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_latency_cdf(metrics: dict, out: Path) -> None:
    runs = metrics["runs"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    # For CDF we approximate with mean/p50/p95/p99/max (we don't serialize
    # the full distribution to keep metrics.json small).
    for r in runs:
        lat = r["metrics"]["latency_ms"]
        xs = np.array([lat["mean"] * 0.7, lat["p50"], lat["p95"], lat["p99"], lat["max"]])
        ys = np.array([0.10, 0.50, 0.95, 0.99, 1.0])
        ax.step(xs, ys, where="post", label=r["spec"]["label"], linewidth=1.3)
    ax.set_xscale("log")
    ax.set_xlabel("end-to-end latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cross_edge_bytes(metrics: dict, out: Path) -> None:
    runs = metrics["runs"]
    labels = [r["spec"]["label"] for r in runs]
    bytes_ = [r["metrics"]["cross_edge_bytes"] for r in runs]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(np.arange(len(labels)), bytes_, color="#4a8a5c")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("cross-edge bytes")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_scaling(scaling: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    weak = scaling["weak"]
    axes[0].plot([p["num_edges"] for p in weak],
                 [p["metrics"]["throughput_ops"] for p in weak],
                 "o-", color="#3b6aa0")
    axes[0].set_title("Weak scaling")
    axes[0].set_xlabel("# edges (entities scaled proportionally)")
    axes[0].set_ylabel("throughput (ops/s)")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    strong = scaling["strong"]
    base = strong[0]["metrics"]["throughput_ops"]
    axes[1].plot([p["num_edges"] for p in strong],
                 [p["metrics"]["throughput_ops"] / max(1e-6, base) for p in strong],
                 "o-", color="#c25b56", label="speedup")
    axes[1].plot([p["num_edges"] for p in strong],
                 [p["num_edges"] / strong[0]["num_edges"] for p in strong],
                 "--", color="k", label="ideal")
    axes[1].set_title("Strong scaling")
    axes[1].set_xlabel("# edges")
    axes[1].set_ylabel("speedup")
    axes[1].set_xscale("log", base=2)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, linestyle="--")

    ent = scaling["entity"]
    axes[2].plot([p["num_entities"] for p in ent],
                 [p["metrics"]["latency_ms"]["p99"] for p in ent],
                 "o-", color="#4a8a5c")
    axes[2].set_title("Entity scaling")
    axes[2].set_xlabel("# entities")
    axes[2].set_ylabel("p99 latency (ms)")
    axes[2].set_xscale("log")
    axes[2].grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ablation(abl: dict, out: Path) -> None:
    rows = abl["variants"]
    labels = [r["variant"] for r in rows]
    p99   = [r["metrics"]["latency_ms"]["p99"] for r in rows]
    tput  = [r["metrics"]["throughput_ops"] for r in rows]
    ce    = [r["metrics"]["cross_edge_bytes"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0))
    ax[0].bar(labels, p99, color="#c25b56"); ax[0].set_ylabel("p99 latency (ms)")
    ax[1].bar(labels, tput, color="#3b6aa0"); ax[1].set_ylabel("throughput (ops/s)")
    ax[2].bar(labels, ce, color="#4a8a5c"); ax[2].set_ylabel("cross-edge bytes"); ax[2].set_yscale("log")
    for a in ax:
        a.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics",  default=str(MODULE_ROOT / "outputs" / "metrics.json"))
    p.add_argument("--scaling",  default=str(MODULE_ROOT / "outputs" / "scaling" / "scaling.json"))
    p.add_argument("--ablation", default=str(MODULE_ROOT / "outputs" / "ablation" / "ablation.json"))
    p.add_argument("--outdir",   default=str(MODULE_ROOT / "outputs" / "figs"))
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path  = Path(args.metrics)
    scaling_path  = Path(args.scaling)
    ablation_path = Path(args.ablation)

    if metrics_path.exists():
        m = _load(metrics_path)
        fig_throughput_latency(m, outdir / "fig_throughput_p99.pdf")
        fig_latency_cdf(m,        outdir / "fig_latency_cdf.pdf")
        fig_cross_edge_bytes(m,   outdir / "fig_cross_edge_bytes.pdf")
        print(f"[plot] wrote throughput / CDF / cross-edge figures to {outdir}")
    if scaling_path.exists():
        s = _load(scaling_path)
        fig_scaling(s, outdir / "fig_scaling.pdf")
        print(f"[plot] wrote scaling figure to {outdir}")
    if ablation_path.exists():
        a = _load(ablation_path)
        fig_ablation(a, outdir / "fig_ablation.pdf")
        print(f"[plot] wrote ablation figure to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render every SkyShield figure (Fig. 6 - Fig. 9 + latency budget bar) and
aggregate ``outputs/metrics.json`` from the per-experiment JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import MODULE_ROOT


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6 (CDF)


def fig_e2e_cdf(timing: dict, out: Path) -> None:
    samples = np.asarray(timing["end_to_end_samples"], dtype=np.float64)
    if samples.size == 0:
        return
    samples = np.sort(samples)
    ys = np.linspace(0, 1, samples.size, endpoint=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(samples, ys, color="#3b6aa0", linewidth=1.6, label="SkyShield (full)")
    ax.axvline(1500.0, color="#c25b56", linestyle="--", linewidth=1.0,
               label="end-to-end deadline (1.5 s)")
    ax.axvline(1450.0, color="#7a7a7a", linestyle=":", linewidth=1.0,
               label="P99 target (1.45 s)")
    ax.set_xlabel("end-to-end latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, max(1700, samples.max() * 1.05))
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out)


# ---------------------------------------------------------------- Fig 7 (tail)


def fig_tail_vs_load(stress: dict, out: Path) -> None:
    rows = stress["regimes"]
    names = [r["regime"] for r in rows]
    p99 = [r["latency"]["end_to_end"]["p99"] for r in rows]
    miss = [r["metrics"]["deadline_miss_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(names))
    bars = ax.bar(x, p99, color="#c25b56", label="P99 latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("P99 end-to-end latency (ms)")
    ax.axhline(1450.0, color="#3b6aa0", linestyle="--", linewidth=1.0,
               label="P99 target (1.45 s)")
    axR = ax.twinx()
    axR.plot(x, miss, color="#4a8a5c", marker="o", linewidth=1.6,
             label="deadline miss %")
    axR.set_ylabel("deadline miss (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    _save(fig, out)


# ---------------------------------------------------------------- Fig 8 (multi-radar)


def fig_multi_radar(mr: dict, out: Path) -> None:
    rows = mr["rows"]
    by_t: dict[int, list[dict]] = {}
    for r in rows:
        by_t.setdefault(r["num_targets"], []).append(r)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    colors = ["#3b6aa0", "#c25b56", "#4a8a5c", "#9a59b3"]

    for i, (ntgt, runs) in enumerate(sorted(by_t.items())):
        runs = sorted(runs, key=lambda r: r["num_radars"])
        xs = [r["num_radars"] for r in runs]
        p99 = [r["latency_ms"]["end_to_end"]["p99"] for r in runs]
        cov = [r["coverage_pct"] for r in runs]
        c = colors[i % len(colors)]
        axes[0].plot(xs, p99, "o-", color=c, label=f"{ntgt} target(s)")
        axes[1].plot(xs, cov, "s-", color=c, label=f"{ntgt} target(s)")

    axes[0].axhline(1450.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("# radar nodes")
    axes[0].set_ylabel("P99 end-to-end latency (ms)")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("# radar nodes")
    axes[1].set_ylabel("coverage of 300 km^2 (%)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(fontsize=8)
    _save(fig, out)


# ---------------------------------------------------------------- Fig 9 (failure flow)


def fig_failure_flow(safety: dict, out: Path) -> None:
    fams = list(safety.keys())
    succ = [safety[f]["headline"]["mission_success_rate_pct"] for f in fams]
    suppress = [safety[f]["headline"]["suppressed_count"] for f in fams]
    aborts = [safety[f]["headline"]["abort_count"] for f in fams]
    lost = [safety[f]["headline"]["target_lost_count"] for f in fams]

    x = np.arange(len(fams))
    w = 0.20
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x - 1.5 * w, succ, width=w, color="#3b6aa0", label="success %")
    ax.bar(x - 0.5 * w, suppress, width=w, color="#c25b56", label="suppressed")
    ax.bar(x + 0.5 * w, aborts, width=w, color="#4a8a5c", label="aborts")
    ax.bar(x + 1.5 * w, lost, width=w, color="#9a59b3", label="target_lost")
    ax.set_xticks(x)
    ax.set_xticklabels(fams, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("count or %")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    _save(fig, out)


# ---------------------------------------------------------------- latency budget


def fig_latency_budget(timing: dict, out: Path) -> None:
    lat = timing["metrics"]["latency_ms"]
    stages = ["detection", "track_confirm", "fusion", "decision",
              "launch", "interceptor_reaction"]
    means = [lat[s]["mean"] for s in stages]
    p99s = [lat[s]["p99"] for s in stages]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(stages))
    w = 0.38
    ax.bar(x - w / 2, means, width=w, color="#3b6aa0", label="mean")
    ax.bar(x + w / 2, p99s, width=w, color="#c25b56", label="P99")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("per-stage latency (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    _save(fig, out)


# ---------------------------------------------------------------- aggregate


def aggregate_metrics(outdir: Path) -> None:
    field = _load(outdir / "field_replay.json") or {}
    timing = _load(outdir / "timing.json") or {}
    stress = _load(outdir / "stress.json") or {}
    mr = _load(outdir / "multi_radar.json") or {}
    abl = _load(outdir / "ablation.json") or {}
    safety = _load(outdir / "safety.json") or {}

    headline = {}
    if timing:
        h = timing["metrics"]["headline"]
        l = timing["metrics"]["latency_ms"]["end_to_end"]
        headline.update({
            "mission_success_rate_pct": round(h["mission_success_rate_pct"], 2),
            "valid_interception_success_pct": round(h["valid_interception_success_pct"], 2),
            "shot_down_rate_pct": round(h["shot_down_rate_pct"], 2),
            "deadline_miss_pct": round(h["deadline_miss_pct"], 4),
            "false_launch_suppression_pct": round(h["false_launch_suppression_pct"], 4),
            "end_to_end_mean_ms": round(l["mean"], 2),
            "end_to_end_p50_ms": round(l["p50"], 2),
            "end_to_end_p95_ms": round(l["p95"], 2),
            "end_to_end_p99_ms": round(l["p99"], 2),
            "end_to_end_max_ms": round(l["max"], 2),
        })
    if field:
        headline["real_field_success_pct"] = round(
            field["real"]["metrics"]["headline"]["mission_success_rate_pct"], 2
        )
        headline["augmented_field_success_pct"] = round(
            field["augmented"]["metrics"]["headline"]["mission_success_rate_pct"], 2
        )

    out = {
        "headline": headline,
        "field_replay": field,
        "timing": {"headline": timing.get("metrics", {}).get("headline", {}),
                   "latency_ms": timing.get("metrics", {}).get("latency_ms", {})},
        "stress": stress,
        "multi_radar": mr,
        "ablation": abl,
        "safety": safety,
    }
    (outdir / "metrics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"[SkyShield][plot] wrote aggregate {outdir / 'metrics.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(MODULE_ROOT / "outputs"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figs = outdir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    timing = _load(outdir / "timing.json")
    stress = _load(outdir / "stress.json")
    mr = _load(outdir / "multi_radar.json")
    safety = _load(outdir / "safety.json")

    if timing:
        fig_e2e_cdf(timing, figs / "fig_cdf.pdf")
        fig_latency_budget(timing, figs / "fig_latency_budget.pdf")
        print(f"[SkyShield][plot] wrote {figs / 'fig_cdf.pdf'}")
    if stress:
        fig_tail_vs_load(stress, figs / "fig_tail.pdf")
        print(f"[SkyShield][plot] wrote {figs / 'fig_tail.pdf'}")
    if mr:
        fig_multi_radar(mr, figs / "fig_scaling.pdf")
        print(f"[SkyShield][plot] wrote {figs / 'fig_scaling.pdf'}")
    if safety:
        fig_failure_flow(safety, figs / "fig_failure.pdf")
        print(f"[SkyShield][plot] wrote {figs / 'fig_failure.pdf'}")

    aggregate_metrics(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

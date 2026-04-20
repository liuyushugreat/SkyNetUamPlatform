"""Multi-seed aggregation for SkyCert main experiment, ablation, and baselines.

Reruns ``run_experiment``, ``run_ablation``, and ``run_baselines`` with a
configurable list of seeds, then aggregates every numeric field into
``mean``/``std``/``min``/``max`` summaries. The output is written to
``outputs/multi_seed.json`` and consumed by the paper tables and figures.

Usage:
    python -m scripts.run_multi_seed --config configs/default.yaml \
        --seeds 20260417 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from skycert.config import SkyCertConfig
from skycert.utils import ensure_dir, safe_json


def _aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _collect_numeric(
    dicts: list[dict[str, Any]], prefix: str = ""
) -> dict[str, dict[str, float]]:
    """Flatten a list of dicts and aggregate every numeric leaf."""
    out: dict[str, list[float]] = defaultdict(list)

    def walk(d: Any, path: str) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(d, (int, float)):
            out[path].append(float(d))

    for d in dicts:
        walk(d, prefix)
    return {k: _aggregate(v) for k, v in out.items()}


def _run_child(args: list[str]) -> None:
    print(f"[multi-seed] $ {' '.join(args)}", flush=True)
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"child command failed: {' '.join(args)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-seed SkyCert aggregation")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260417, 1, 2, 3, 4],
        help="list of seeds to run (default: 5 seeds)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="only aggregate, do not relaunch per-seed experiments",
    )
    args = parser.parse_args(argv)

    config = SkyCertConfig.load(args.config)
    out_dir = ensure_dir(config.output_dir)

    per_seed_metrics: list[dict[str, Any]] = []
    per_seed_ablation: list[dict[str, Any]] = []
    per_seed_baselines: list[dict[str, Any]] = []

    for seed in args.seeds:
        tag = f"seed_{seed}"
        mfile = f"metrics_{tag}.json"
        afile = f"ablation_{tag}.json"
        bfile = f"baselines_{tag}.json"

        if not args.skip_run:
            _run_child([
                sys.executable, "-m", "scripts.run_experiment",
                "--config", args.config,
                "--seed", str(seed),
                "--output-name", mfile,
                "--audit-dir-name", f"audit_{tag}",
            ])
            _run_child([
                sys.executable, "-m", "scripts.run_ablation",
                "--config", args.config,
                "--seed", str(seed),
                "--output-name", afile,
                "--audit-dir-name", f"audit_ablation_{tag}",
            ])
            _run_child([
                sys.executable, "-m", "scripts.run_baselines",
                "--config", args.config,
                "--seed", str(seed),
                "--output-name", bfile,
            ])

        with open(out_dir / mfile, "r", encoding="utf-8") as fh:
            per_seed_metrics.append(json.load(fh))
        with open(out_dir / afile, "r", encoding="utf-8") as fh:
            per_seed_ablation.append(json.load(fh))
        with open(out_dir / bfile, "r", encoding="utf-8") as fh:
            per_seed_baselines.append(json.load(fh))

    # Aggregate main experiment (per scenario).
    scenarios = [r["threat"]["name"] for r in per_seed_metrics[0]["runs"]]
    main_agg: dict[str, Any] = {}
    for sc in scenarios:
        runs = [
            next(r for r in seed_run["runs"] if r["threat"]["name"] == sc)
            for seed_run in per_seed_metrics
        ]
        main_agg[sc] = _collect_numeric(runs)

    # Aggregate ablation (per variant).
    ablation_variants = [r["variant"] for r in per_seed_ablation[0]["runs"]]
    abl_agg: dict[str, Any] = {}
    for v in ablation_variants:
        runs = [
            next(r for r in seed_run["runs"] if r["variant"] == v)
            for seed_run in per_seed_ablation
        ]
        abl_agg[v] = _collect_numeric(runs)

    # Aggregate baselines (per method).
    methods = [b["method"] for b in per_seed_baselines[0]["baselines"]]
    base_agg: dict[str, Any] = {}
    for m in methods:
        runs = [
            next(b for b in seed_run["baselines"] if b["method"] == m)
            for seed_run in per_seed_baselines
        ]
        base_agg[m] = _collect_numeric(runs)

    payload = {
        "seeds": list(args.seeds),
        "main": main_agg,
        "ablation": abl_agg,
        "baselines": base_agg,
    }

    out_path = out_dir / "multi_seed.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(safe_json(payload), fh, indent=2)
    print(f"[multi-seed] wrote {out_path}")


if __name__ == "__main__":
    main()

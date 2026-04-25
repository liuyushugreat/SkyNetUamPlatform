"""Multi-seed main-comparison driver.

Runs the eight baseline configurations across N seeds and aggregates
mean / std-dev for throughput, latency percentiles, cross-edge bytes,
and state-tier hit ratio.  Writes ``outputs/multiseed.json`` that the
paper's Table 1 consumes.

Example::

    python scripts/run_multiseed.py --config configs/default.yaml \
        --seeds 20260928 20260929 20260930 20260931 20260932 \
        --out outputs/multiseed.json
"""

from __future__ import annotations

import argparse
import math
import statistics as _st
import sys
from pathlib import Path
from typing import Any

from _common import MODULE_ROOT, default_config_path

sys.path.insert(0, str(MODULE_ROOT))

from skygrid.config import SkyGridConfig
from skygrid.runtime import RuntimeConfig, SkyGridRuntime
from skygrid.utils import dump_json
from skygrid.workload import CityScaleWorkload
from skygrid.workload.dag import TaskDAG


# The reviewer-visible set of baselines.  ``loc_aware`` is the new
# state-tier-blind COP baseline; it sits between ``ldg+cop`` and the
# full SkyGrid and isolates the contribution of the state-tier term.
BASELINES: list[dict[str, Any]] = [
    {"label": "cloud-only",    "partition": "hash", "placement": "all_cloud", "pipeline": "abp"},
    {"label": "edge-only",     "partition": "hash", "placement": "all_edge",  "pipeline": "abp"},
    {"label": "hash+static",   "partition": "hash",   "placement": "static",    "pipeline": "abp"},
    {"label": "random+static", "partition": "random", "placement": "static",    "pipeline": "abp"},
    {"label": "ldg+static",    "partition": "ldg",    "placement": "static",    "pipeline": "abp"},
    {"label": "ldg+cop",       "partition": "ldg",    "placement": "cop",       "pipeline": "abp"},
    {"label": "ldg+locaware",  "partition": "ldg",    "placement": "loc_aware", "pipeline": "abp"},
    {"label": "skygrid",       "partition": "stp",    "placement": "cop",       "pipeline": "abp"},
]


def _pctile_key(d: dict[str, float], k: str) -> float:
    return float(d.get(k, float("nan")))


def _agg(xs: list[float]) -> dict[str, float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return {"mean": float("nan"), "stdev": 0.0, "n": 0}
    mean = _st.fmean(xs)
    stdev = _st.pstdev(xs) if len(xs) > 1 else 0.0
    return {"mean": mean, "stdev": stdev, "n": len(xs)}


def _run_one(cfg: SkyGridConfig, spec: dict[str, Any]) -> dict:
    cfg.partition.method = spec["partition"]
    cfg.placement.method = spec["placement"]
    cfg.pipeline.method  = spec["pipeline"]
    dag = TaskDAG.from_config(cfg.dag)
    w = CityScaleWorkload(
        cfg.workload, dag, seed=cfg.seed,
        cells_per_side=cfg.partition.grid.cells_per_side,
    )
    rt = SkyGridRuntime(cfg, RuntimeConfig(label=spec["label"]))
    m = rt.run(w)
    return m.to_json()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(default_config_path()))
    p.add_argument("--seeds", nargs="+", type=int, default=[
        20260928, 20260929, 20260930, 20260931, 20260932,
    ])
    p.add_argument("--out", default=None, help="output JSON path")
    args = p.parse_args()

    base_cfg_path = Path(args.config)
    out_path = Path(args.out) if args.out else (
        MODULE_ROOT / "outputs" / "multiseed.json"
    )

    # per-baseline per-seed raw metrics
    per_baseline: dict[str, list[dict]] = {b["label"]: [] for b in BASELINES}

    for seed in args.seeds:
        print(f"\n[multiseed] ==== seed={seed} ====", flush=True)
        for spec in BASELINES:
            cfg = SkyGridConfig.load(base_cfg_path)
            cfg.seed = int(seed)
            m = _run_one(cfg, spec)
            per_baseline[spec["label"]].append(m)
            print(
                f"[multiseed] {spec['label']:<16} seed={seed} "
                f"p99={_pctile_key(m['latency_ms'], 'p99'):.2f}ms "
                f"xedge={m['cross_edge_bytes']/1e6:.1f}MB "
                f"tput={m['throughput_ops']:.1f}ops/s",
                flush=True,
            )

    # aggregate
    summary: list[dict[str, Any]] = []
    for label, runs in per_baseline.items():
        if not runs:
            continue
        agg: dict[str, Any] = {"label": label, "n_seeds": len(runs)}
        for key in ("p50", "p95", "p99"):
            agg[f"latency_{key}_ms"] = _agg(
                [_pctile_key(r["latency_ms"], key) for r in runs]
            )
        agg["throughput_ops"]       = _agg([r["throughput_ops"]       for r in runs])
        agg["cross_edge_bytes"]     = _agg([r["cross_edge_bytes"]     for r in runs])
        agg["state_tier_hit_ratio"] = _agg([r["state_tier_hit_ratio"] for r in runs])
        agg["state_tier_avg_ms"]    = _agg([r["state_tier_avg_ms"]    for r in runs])
        agg["cloud_util"]           = _agg([r["cloud_util"]           for r in runs])
        # mean of per-edge avg util across seeds
        agg["edge_util_avg"]        = _agg(
            [sum(r["edge_util"]) / max(1, len(r["edge_util"])) for r in runs]
        )
        agg["partition_edge_cut"] = _agg(
            [float(r["partition_info"].get("edge_cut", 0.0)) for r in runs]
        )
        agg["partition_load_imbalance"] = _agg(
            [float(r["partition_info"].get("load_imbalance", 1.0)) for r in runs]
        )
        summary.append(agg)

    dump_json(out_path, {
        "config_path": str(base_cfg_path),
        "seeds": list(args.seeds),
        "summary": summary,
        "raw": {lbl: runs for lbl, runs in per_baseline.items()},
    })
    print(f"\n[multiseed] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

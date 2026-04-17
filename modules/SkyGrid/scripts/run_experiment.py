"""Main experiment: SkyGrid vs 6 baselines on the ``M`` regime.

Produces ``outputs/metrics.json`` with one row per (partition, placement,
pipeline) configuration and the measured throughput / latency / p95 /
p99 / cross-edge comm / utilization.
"""

from __future__ import annotations

import argparse
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


BASELINES: list[dict[str, Any]] = [
    {"label": "cloud-only",      "partition": "hash", "placement": "all_cloud", "pipeline": "abp"},
    {"label": "edge-only",       "partition": "hash", "placement": "all_edge",  "pipeline": "abp"},
    {"label": "hash+static",     "partition": "hash",   "placement": "static", "pipeline": "abp"},
    {"label": "random+static",   "partition": "random", "placement": "static", "pipeline": "abp"},
    {"label": "ldg+static",      "partition": "ldg",    "placement": "static", "pipeline": "abp"},
    {"label": "ldg+cop",         "partition": "ldg",    "placement": "cop",    "pipeline": "abp"},
    {"label": "skygrid (stp+cop+abp)", "partition": "stp", "placement": "cop", "pipeline": "abp"},
]


def run_one(cfg: SkyGridConfig, spec: dict[str, Any]) -> dict:
    cfg.partition.method = spec["partition"]
    cfg.placement.method = spec["placement"]
    cfg.pipeline.method = spec["pipeline"]
    dag = TaskDAG.from_config(cfg.dag)
    w = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed,
                          cells_per_side=cfg.partition.grid.cells_per_side)
    rt = SkyGridRuntime(cfg, RuntimeConfig(label=spec["label"]))
    m = rt.run(w)
    return {"spec": spec, "metrics": m.to_json()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(default_config_path()))
    p.add_argument("--output", default=None,
                   help="output path for metrics.json (defaults to outputs/metrics.json)")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = SkyGridConfig.load(cfg_path)
    out_path = Path(args.output) if args.output else (
        MODULE_ROOT / cfg.output_dir / "metrics.json"
    )

    runs = []
    for spec in BASELINES:
        print(f"[SkyGrid] running {spec['label']:<40} "
              f"(partition={spec['partition']}, placement={spec['placement']}, "
              f"pipeline={spec['pipeline']})", flush=True)
        runs.append(run_one(cfg, spec))

    dump_json(out_path, {
        "config": {
            "num_entities": cfg.workload.num_entities,
            "duration_s":   cfg.workload.duration_s,
            "num_edges":    cfg.fabric.edge.num_nodes,
            "seed":         cfg.seed,
        },
        "runs": runs,
    })
    print(f"[SkyGrid] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

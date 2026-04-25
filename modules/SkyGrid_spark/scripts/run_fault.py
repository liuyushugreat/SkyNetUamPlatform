"""Graceful-degradation / fault-injection experiment.

Replays the M-regime workload on a 4-edge fabric where one or more
edges are in a degraded-compute state (simulated via a reduced
``tflops_per_node``).  The goal is to quantify how SkyGrid's
locality-aware placement tolerates partial outages versus static
placement baselines.

Each row of the output JSON records a (config, fault_level) pair with
the usual latency / throughput / cross-edge metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import MODULE_ROOT, default_config_path

sys.path.insert(0, str(MODULE_ROOT))

from skygrid.config import SkyGridConfig
from skygrid.runtime import RuntimeConfig, SkyGridRuntime
from skygrid.utils import dump_json
from skygrid.workload import CityScaleWorkload
from skygrid.workload.dag import TaskDAG


# fault profile name -> per-edge TFLOPS vector
PROFILES = {
    "healthy":            [1.0, 1.0, 1.0, 1.0],
    "one-edge-degraded":  [1.0, 1.0, 1.0, 0.3],
    "two-edges-degraded": [1.0, 1.0, 0.5, 0.3],
    "one-edge-failed":    [1.0, 1.0, 1.0, 0.05],   # ~ effectively offline
}

BASELINES = [
    {"label": "ldg+static",   "partition": "ldg", "placement": "static",    "pipeline": "abp"},
    {"label": "ldg+cop",      "partition": "ldg", "placement": "cop",       "pipeline": "abp"},
    {"label": "ldg+locaware", "partition": "ldg", "placement": "loc_aware", "pipeline": "abp"},
    {"label": "skygrid",      "partition": "stp", "placement": "cop",       "pipeline": "abp"},
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(default_config_path()))
    p.add_argument("--out",    default=None)
    args = p.parse_args()

    base_cfg_path = Path(args.config)
    out_path = Path(args.out) if args.out else (
        MODULE_ROOT / "outputs" / "fault" / "fault.json"
    )

    rows = []
    for profile_name, tflops_vec in PROFILES.items():
        for spec in BASELINES:
            cfg = SkyGridConfig.load(base_cfg_path)
            cfg.fabric.edge.tflops_per_node = list(tflops_vec)
            cfg.partition.method = spec["partition"]
            cfg.placement.method = spec["placement"]
            cfg.pipeline.method  = spec["pipeline"]

            dag = TaskDAG.from_config(cfg.dag)
            w = CityScaleWorkload(
                cfg.workload, dag, seed=cfg.seed,
                cells_per_side=cfg.partition.grid.cells_per_side,
            )
            rt = SkyGridRuntime(cfg, RuntimeConfig(label=f"{spec['label']}@{profile_name}"))
            m = rt.run(w)
            rows.append({
                "profile": profile_name,
                "tflops_per_node": tflops_vec,
                "spec": spec,
                "metrics": m.to_json(),
            })
            print(
                f"[fault] {profile_name:<20} {spec['label']:<14} "
                f"p99={m.latency_ms['p99']:.2f}ms "
                f"xedge={m.cross_edge_bytes/1e6:.1f}MB",
                flush=True,
            )

    dump_json(out_path, {"profiles": PROFILES, "rows": rows})
    print(f"\n[fault] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

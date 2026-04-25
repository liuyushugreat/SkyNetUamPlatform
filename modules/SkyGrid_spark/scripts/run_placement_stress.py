"""Controlled placement stress for state-tier awareness.

This is a small cost-model experiment, not an end-to-end simulation. It
constructs a regime where cloud compute and transfer are attractive enough
that a state-tier-blind placer moves symbolic stages to the cloud. COP-H then
keeps the state-heavy tail on the edge once cold-state latency is included.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import MODULE_ROOT, default_config_path

sys.path.insert(0, str(MODULE_ROOT))

from skygrid.config import SkyGridConfig
from skygrid.placement import COPSolver, CostModel, LocAwareSolver
from skygrid.utils import dump_json
from skygrid.workload.dag import TaskDAG


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--out", default=str(MODULE_ROOT / "outputs" / "placement_stress.json"))
    args = parser.parse_args()

    cfg = SkyGridConfig.load(args.config)
    cfg.fabric.cloud.tflops = 10.0
    cfg.fabric.network.edge_cloud_latency_ms = 0.05
    cfg.fabric.network.edge_cloud_bw_gbps = 100.0
    cfg.fabric.state_tier.hot_hit_rate = 0.20
    cfg.fabric.state_tier.warm_hit_rate = 0.20

    for op in cfg.dag.ops:
        if op.kind == "symbolic":
            op.prefers = "cloud"
            op.state_refs = 8
            op.cost_flops = float(op.cost_flops) * 100_000

    dag = TaskDAG.from_config(cfg.dag)
    cost_model = CostModel(cfg.fabric)
    locaware_model = CostModel(cfg.fabric, override_state_tier_enabled=False)
    locaware = LocAwareSolver(dag, cost_model).solve()
    cop = COPSolver(dag, cost_model).solve()

    result = {
        "setting": {
            "cloud_tflops": cfg.fabric.cloud.tflops,
            "edge_cloud_latency_ms": cfg.fabric.network.edge_cloud_latency_ms,
            "edge_cloud_bw_gbps": cfg.fabric.network.edge_cloud_bw_gbps,
            "symbolic_state_refs": 8,
            "symbolic_flops_multiplier": 100_000,
            "hot_hit_rate": cfg.fabric.state_tier.hot_hit_rate,
            "warm_hit_rate": cfg.fabric.state_tier.warm_hit_rate,
        },
        "locaware": {
            "placement": locaware,
            "critical_path_ms_with_state": cost_model.total_cost(dag, locaware),
            "critical_path_ms_without_state": locaware_model.total_cost(dag, locaware),
        },
        "cop_h": {
            "placement": cop,
            "critical_path_ms_with_state": cost_model.total_cost(dag, cop),
        },
    }
    dump_json(Path(args.out), result)
    print(
        "[placement-stress] "
        f"LocAware={result['locaware']['critical_path_ms_with_state']:.2f}ms "
        f"COP-H={result['cop_h']['critical_path_ms_with_state']:.2f}ms",
        flush=True,
    )
    print(f"[placement-stress] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

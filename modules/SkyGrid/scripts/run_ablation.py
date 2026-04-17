"""Ablation sweep: ``full``, ``-STP``, ``-COP``, ``-ABP``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from _common import MODULE_ROOT, ablation_config_path, default_config_path

sys.path.insert(0, str(MODULE_ROOT))

from skygrid.config import SkyGridConfig
from skygrid.runtime import RuntimeConfig, SkyGridRuntime
from skygrid.utils import dump_json
from skygrid.workload import CityScaleWorkload
from skygrid.workload.dag import TaskDAG


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default=str(default_config_path()))
    p.add_argument("--ablation", default=str(ablation_config_path()))
    p.add_argument("--output",  default=None)
    args = p.parse_args()

    cfg = SkyGridConfig.load(args.config)
    ablation_cfg = yaml.safe_load(Path(args.ablation).read_text(encoding="utf-8"))
    out_path = Path(args.output) if args.output else (
        MODULE_ROOT / ablation_cfg.get("output_dir", "outputs/ablation") / "ablation.json"
    )

    rows = []
    for name, spec in ablation_cfg["variants"].items():
        cfg.partition.method = spec["partition"]
        cfg.placement.method = spec["placement"]
        cfg.pipeline.method  = spec["pipeline"]
        dag = TaskDAG.from_config(cfg.dag)
        w = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed,
                              cells_per_side=cfg.partition.grid.cells_per_side)
        rt = SkyGridRuntime(cfg, RuntimeConfig(label=name))
        m = rt.run(w)
        rows.append({"variant": name, "spec": spec, "metrics": m.to_json()})
        print(f"[ablation] {name:<10} p99={m.latency_ms['p99']:.2f}ms "
              f"tput={m.throughput_ops:.1f}ops/s cross_edge={m.cross_edge_bytes:.0f}B",
              flush=True)

    dump_json(out_path, {
        "config": {
            "num_entities": cfg.workload.num_entities,
            "duration_s":   cfg.workload.duration_s,
            "num_edges":    cfg.fabric.edge.num_nodes,
            "seed":         cfg.seed,
        },
        "variants": rows,
    })
    print(f"[ablation] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

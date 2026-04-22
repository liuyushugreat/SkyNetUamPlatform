"""Weak / strong / entity scaling sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from _common import MODULE_ROOT, default_config_path, scaling_config_path

sys.path.insert(0, str(MODULE_ROOT))

from skygrid.config import SkyGridConfig
from skygrid.runtime import RuntimeConfig, SkyGridRuntime
from skygrid.utils import dump_json
from skygrid.workload import CityScaleWorkload
from skygrid.workload.dag import TaskDAG


def _run_point(cfg: SkyGridConfig, num_entities: int, num_edges: int,
               duration_s: float, label: str) -> dict:
    cfg.workload.num_entities = int(num_entities)
    cfg.fabric.edge.num_nodes = int(num_edges)
    cfg.workload.duration_s = float(duration_s)
    dag = TaskDAG.from_config(cfg.dag)
    w = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed,
                          cells_per_side=cfg.partition.grid.cells_per_side)
    rt = SkyGridRuntime(cfg, RuntimeConfig(label=label))
    m = rt.run(w)
    return {
        "label": label,
        "num_entities": num_entities,
        "num_edges": num_edges,
        "duration_s": duration_s,
        "metrics": m.to_json(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default=str(default_config_path()))
    p.add_argument("--scaling", default=str(scaling_config_path()))
    p.add_argument("--output",  default=None)
    args = p.parse_args()

    base_cfg = SkyGridConfig.load(args.config)
    sc = yaml.safe_load(Path(args.scaling).read_text(encoding="utf-8"))
    duration = float(sc.get("duration_s", 20.0))
    out_dir  = MODULE_ROOT / sc.get("output_dir", "outputs/scaling")
    out_path = Path(args.output) if args.output else (out_dir / "scaling.json")

    results: dict = {"weak": [], "strong": [], "entity": []}

    for pt in sc["sweeps"]["weak"]:
        r = _run_point(SkyGridConfig.from_dict(base_cfg.__dict__.copy() | {
            "workload": {**base_cfg.workload.__dict__, "num_entities": pt["num_entities"]},
        }) if False else base_cfg,  # reuse same cfg
                       pt["num_entities"], pt["num_edges"], duration,
                       f"weak_{pt['num_entities']}_{pt['num_edges']}")
        print(f"[scaling/weak] {r['label']:<30} "
              f"tput={r['metrics']['throughput_ops']:.1f}ops/s",
              flush=True)
        results["weak"].append(r)

    strong = sc["sweeps"]["strong"]
    for k in strong["num_edges"]:
        r = _run_point(base_cfg, strong["num_entities"], k, duration,
                       f"strong_{k}")
        print(f"[scaling/strong] {r['label']:<30} "
              f"tput={r['metrics']['throughput_ops']:.1f}ops/s",
              flush=True)
        results["strong"].append(r)

    ent = sc["sweeps"]["entity_scale"]
    for n in ent["num_entities"]:
        r = _run_point(base_cfg, n, ent["num_edges"], duration,
                       f"entity_{n}")
        print(f"[scaling/entity] {r['label']:<30} "
              f"tput={r['metrics']['throughput_ops']:.1f}ops/s",
              flush=True)
        results["entity"].append(r)

    dump_json(out_path, results)
    print(f"[scaling] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

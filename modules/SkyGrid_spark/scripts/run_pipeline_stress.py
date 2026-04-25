"""Pipeline stress test for ABP versus synchronous execution.

The M-regime ablation shows that a synchronous pipeline can have lower
short-tail latency when the fabric is not saturated. This driver raises the
event rate and burstiness to exercise the backpressure regime where ABP's
bounded overlap matters.
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


def _run_one(cfg: SkyGridConfig, label: str, pipeline: str) -> dict:
    cfg.partition.method = "stp"
    cfg.placement.method = "cop"
    cfg.pipeline.method = pipeline

    dag = TaskDAG.from_config(cfg.dag)
    workload = CityScaleWorkload(
        cfg.workload,
        dag,
        seed=cfg.seed,
        cells_per_side=cfg.partition.grid.cells_per_side,
    )
    runtime = SkyGridRuntime(cfg, RuntimeConfig(label=label))
    metrics = runtime.run(workload)
    return {"label": label, "pipeline": pipeline, "metrics": metrics.to_json()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--out", default=str(MODULE_ROOT / "outputs" / "burst_pipeline.json"))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--event-rate", type=float, default=3.0)
    parser.add_argument("--jitter-cov", type=float, default=1.25)
    args = parser.parse_args()

    rows = []
    for label, pipeline in (("burst_abp", "abp"), ("burst_sync", "sync")):
        cfg = SkyGridConfig.load(args.config)
        cfg.workload.duration_s = float(args.duration)
        cfg.workload.event_rate_per_entity_s = float(args.event_rate)
        cfg.workload.jitter.cov = float(args.jitter_cov)
        row = _run_one(cfg, label, pipeline)
        rows.append(row)
        lat = row["metrics"]["latency_ms"]
        print(
            f"[pipeline-stress] {label:<10} p95={lat['p95']:.2f}ms "
            f"p99={lat['p99']:.2f}ms throughput={row['metrics']['throughput_ops']:.1f}ops/s",
            flush=True,
        )

    dump_json(
        Path(args.out),
        {
            "config": {
                "num_entities": cfg.workload.num_entities,
                "duration_s": float(args.duration),
                "event_rate_per_entity_s": float(args.event_rate),
                "jitter_cov": float(args.jitter_cov),
            },
            "runs": rows,
        },
    )
    print(f"[pipeline-stress] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

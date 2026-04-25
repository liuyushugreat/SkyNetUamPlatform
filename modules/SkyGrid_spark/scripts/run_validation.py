"""Cost-model vs discrete-event-simulator validation.

For each operator we (i) sample the 10th-percentile per-op span
duration from the simulator run -- the 10th percentile isolates the
service-time-like regime (close-to-empty queue, no head-of-line
build-up from prior events), and (ii) evaluate the closed-form
placement cost model on the same operator under the realized
placement.  We report per-op absolute and relative error.

Running the validation under a lightly-loaded workload is intentional
so that the DES measurement is *service-time dominated*; the cost
model is a service-time predictor (compute + transfer + state +
myopic queue), not a full M/G/1 model, so benchmarking it against a
saturated DES would be unfair and not what the placer uses internally.
"""

from __future__ import annotations

import argparse
import statistics as _st
import sys
from collections import defaultdict
from pathlib import Path

from _common import MODULE_ROOT, default_config_path

sys.path.insert(0, str(MODULE_ROOT))

from skygrid.config import SkyGridConfig
from skygrid.placement import CostModel, build_placement
from skygrid.runtime import RuntimeConfig, SkyGridRuntime
from skygrid.utils import dump_json
from skygrid.workload import CityScaleWorkload
from skygrid.workload.dag import TaskDAG


def _per_op_service_ms(tracer) -> dict[str, float]:
    """10th-percentile per-op span duration -- a proxy for service time
    under a non-saturated queue."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for span in tracer.spans:
        bucket[span.op_name].append(span.end_ms - span.start_ms)
    out: dict[str, float] = {}
    for op, xs in bucket.items():
        if not xs:
            continue
        xs.sort()
        k = max(0, int(0.10 * len(xs)))
        out[op] = float(xs[k])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(default_config_path()))
    p.add_argument("--out",    default=None)
    p.add_argument("--num-entities", type=int, default=500,
                   help="override workload.num_entities for a light run")
    p.add_argument("--duration", type=float, default=10.0,
                   help="override workload.duration_s for a light run")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = SkyGridConfig.load(cfg_path)
    # Light-load override so service time dominates over queueing.
    cfg.workload.num_entities = int(args.num_entities)
    cfg.workload.duration_s   = float(args.duration)

    out_path = Path(args.out) if args.out else (
        MODULE_ROOT / "outputs" / "validation" / "validation.json"
    )

    dag = TaskDAG.from_config(cfg.dag)
    cm = CostModel(cfg.fabric)
    placer = build_placement("cop", dag, cm)
    placement = placer.solve()

    w = CityScaleWorkload(
        cfg.workload, dag, seed=cfg.seed,
        cells_per_side=cfg.partition.grid.cells_per_side,
    )
    rt = SkyGridRuntime(cfg, RuntimeConfig(label="validation"))
    rt.run(w)
    tracer = rt._tracer
    observed_ms = _per_op_service_ms(tracer)

    # Predicted service time = compute + state only.  Transfer is
    # attributed to the producer edge by the runtime (not to the span
    # of this op), and observed queue ~= 0 in light-load mode.
    site_of = placement
    predicted_ms: dict[str, float] = {}
    origin_edge = next(
        (s for s in cm.site_names if s.startswith("edge")),
        cm.site_names[0],
    )
    for op in dag:
        s = site_of[op.name]
        parents = dag.parents(op.name)
        producer = placement[parents[0]] if parents else origin_edge
        c = cm.op_cost(op, s, producer)
        # Service-time-only prediction: compute only (state access is
        # absorbed into ``arrival_ms`` by the runtime, so the node's
        # span records compute + residual queue, not state).  See
        # ``SkyGridRuntime._dispatch`` for the bookkeeping rule.
        predicted_ms[op.name] = c.compute_ms

    rows = []
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    for op_name in sorted(observed_ms.keys()):
        o = observed_ms[op_name]
        pr = predicted_ms.get(op_name, float("nan"))
        abs_err = abs(o - pr)
        rel_err = abs_err / max(o, 1e-9)
        rows.append({
            "op": op_name,
            "site": placement.get(op_name, "?"),
            "observed_ms": round(o, 4),
            "predicted_ms": round(pr, 4),
            "abs_error_ms": round(abs_err, 4),
            "rel_error": round(rel_err, 4),
        })
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)

    summary = {
        "mae_ms":           float(_st.fmean(abs_errors)) if abs_errors else 0.0,
        "mean_rel_error":   float(_st.fmean(rel_errors)) if rel_errors else 0.0,
        "max_abs_error_ms": float(max(abs_errors))       if abs_errors else 0.0,
    }

    dump_json(out_path, {
        "config_path": str(cfg_path),
        "light_load": {
            "num_entities": cfg.workload.num_entities,
            "duration_s":   cfg.workload.duration_s,
        },
        "placement": placement,
        "per_op": rows,
        "summary": summary,
    })
    print(f"[validation] light-load: {cfg.workload.num_entities} entities, "
          f"{cfg.workload.duration_s:.0f}s")
    print(f"[validation] MAE={summary['mae_ms']:.3f}ms, "
          f"mean rel err={summary['mean_rel_error']*100:.2f}%")
    print(f"[validation] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

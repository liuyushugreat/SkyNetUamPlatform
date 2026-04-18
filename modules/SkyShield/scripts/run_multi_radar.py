"""E4: Multi-radar urban deployment sweep."""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml
import numpy as np

from skyshield.config import load_config
from skyshield.runtime.engine import SkyShieldRuntime
from skyshield.workload import generate

from scripts._common import arg_parser, ensure_outputs, write_json


def _uniform_radar_grid(n: int, bbox):
    """Place ``n`` radars on a near-uniform grid inside ``bbox``."""
    x0, y0, x1, y1 = bbox
    cols = max(1, int(round(math.sqrt(n * (x1 - x0) / max(1e-3, (y1 - y0))))))
    rows = int(math.ceil(n / cols))
    xs = np.linspace(x0 + 1.0, x1 - 1.0, cols)
    ys = np.linspace(y0 + 1.0, y1 - 1.0, rows)
    placements = []
    for j in range(rows):
        for i in range(cols):
            if len(placements) >= n:
                break
            placements.append([float(xs[i]), float(ys[j])])
    return placements[:n]


def main() -> None:
    parser = arg_parser("SkyShield E4: multi-radar deployment sweep.")
    parser.add_argument("--duration", type=float, default=180.0)
    args = parser.parse_args()

    out_dir = ensure_outputs(args.out)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_path = Path(args.config).parent / raw["base"]

    sweep = raw["sweep"]
    seeds = json.loads(
        Path("data/augmented_seeds.json").read_text(encoding="utf-8")
    )["seeds"]
    base_seed = seeds["multi_radar_sweep"]

    rows = []
    base_cfg = load_config(str(base_path))
    for nr in sweep["radar_counts"]:
        placement = _uniform_radar_grid(nr, base_cfg.city.bbox_km)
        for tc in sweep["target_concurrency"]:
            per_cell_p99 = []
            per_cell_miss = []
            per_cell_handoff = []
            per_cell_mission = []
            for off in sweep["seed_offsets"]:
                cfg = base_cfg.with_overrides({
                    "radars.count": nr,
                    "radars.placement": placement,
                })
                sc = generate(cfg, duration_s=args.duration,
                              concurrency=tc, seed=base_seed + off)
                rt = SkyShieldRuntime(cfg, config_path=str(args.config))
                rep = rt.run(sc)
                s = rep.metrics.summary()
                per_cell_p99.append(s["latency_ms"]["p99"])
                per_cell_miss.append(s["deadline_miss_ratio"])
                per_cell_handoff.append(s["radar_handoff_latency_ms"]["p95"])
                per_cell_mission.append(s["mission_success_rate"])
            rows.append({
                "num_radars": nr,
                "target_concurrency": tc,
                "mission_success_mean": float(np.mean(per_cell_mission)),
                "p99_latency_ms_mean": float(np.mean(per_cell_p99)),
                "deadline_miss_mean": float(np.mean(per_cell_miss)),
                "handoff_p95_ms_mean": float(np.mean(per_cell_handoff)),
                "num_seeds": len(sweep["seed_offsets"]),
            })
            print(f"[E4] radars={nr} conc={tc} p99={rows[-1]['p99_latency_ms_mean']:.1f} "
                  f"miss={rows[-1]['deadline_miss_mean']:.4f}")

    write_json({"config_path": str(args.config), "rows": rows},
               out_dir / "multi_radar.json")


if __name__ == "__main__":
    main()

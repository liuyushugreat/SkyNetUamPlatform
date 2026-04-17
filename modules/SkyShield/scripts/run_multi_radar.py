"""E4: multi-radar urban deployment sweep over 300 km^2."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from _common import MODULE_ROOT, augmented_scenarios, real_scenarios

from skyshield.config import SkyShieldConfig
from skyshield.runtime import RuntimeOptions, SkyShieldRuntime
from skyshield.utils import dump_json


def coverage_pct(num_radars: int, area_km2: float, range_km: float) -> float:
    """Crude coverage estimate: fraction of grid cells within range of >=1 radar."""
    import numpy as np
    side_km = float(np.sqrt(area_km2))
    n = 30
    xs = np.linspace(-side_km / 2, side_km / 2, n)
    ys = np.linspace(-side_km / 2, side_km / 2, n)
    pts = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T

    half = side_km * 1000.0 / 2
    if num_radars <= 0:
        return 0.0
    if num_radars == 1:
        rad_pos = np.array([[0.0, 0.0]])
    elif num_radars == 2:
        rad_pos = np.array([[-half * 0.5, -half * 0.5], [half * 0.5, half * 0.5]])
    elif num_radars == 4:
        rad_pos = np.array([
            [-half * 0.5, -half * 0.5],
            [half * 0.5, -half * 0.5],
            [-half * 0.5, half * 0.5],
            [half * 0.5, half * 0.5],
        ])
    else:
        radius = half * 0.55
        rad_pos = np.array([
            [radius * np.cos(2 * np.pi * k / num_radars),
             radius * np.sin(2 * np.pi * k / num_radars)]
            for k in range(num_radars)
        ])

    pts_m = pts * 1000.0
    covered = 0
    for p in pts_m:
        d = np.linalg.norm(rad_pos - p, axis=1)
        if (d <= range_km * 1000.0).any():
            covered += 1
    return 100.0 * covered / pts.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MODULE_ROOT / "configs" / "multi_radar.yaml"))
    ap.add_argument("--base-config", default=str(MODULE_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(MODULE_ROOT / "outputs" / "multi_radar.json"))
    args = ap.parse_args()

    base = SkyShieldConfig.load(Path(args.base_config))
    sweep = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["sweep"]

    base_scens = real_scenarios() + augmented_scenarios(rng_seed=base.seed)

    rows = []
    for nrad in sweep["radar_counts"]:
        for ntgt in sweep["target_counts"]:
            cfg = SkyShieldConfig.load(Path(args.base_config))
            # Heavier load with more concurrent targets; the prioritization
            # and deadline scheduler partially absorb this.
            load = min(0.90, 0.55 + 0.06 * (ntgt - 1))
            opts = RuntimeOptions(
                label=f"r{nrad}_t{ntgt}",
                radar_count_override=nrad,
                target_count=ntgt,
                load_scale=load,
            )
            rt = SkyShieldRuntime(cfg, opts)
            # Inflate the workload by ntgt to simulate concurrent targets.
            scens = []
            for k in range(ntgt):
                for s in base_scens[: max(8, len(base_scens) // 2)]:
                    scens.append(
                        type(s)(
                            sortie_id=s.sortie_id + 1000 * (k + 1),
                            test_type=s.test_type,
                            target_takeoff_t=s.target_takeoff_t,
                            target_speed_kmh=s.target_speed_kmh,
                            target_height_m=s.target_height_m,
                            interceptor_takeoff_t=s.interceptor_takeoff_t,
                            is_real=s.is_real,
                            expected_outcome=s.expected_outcome,
                            forced_abort=s.forced_abort,
                            forced_lost_lock=s.forced_lost_lock,
                            target_maneuver_g=s.target_maneuver_g,
                            spawn_distance_m=s.spawn_distance_m,
                        )
                    )
            rt.run(scens)
            j = rt.metrics.to_json()
            cov = coverage_pct(nrad, cfg.scenario.area_km2, cfg.radar.range_km)
            rows.append({
                "num_radars": nrad,
                "num_targets": ntgt,
                "coverage_pct": cov,
                "headline": j["headline"],
                "latency_ms": j["latency_ms"],
            })
            print(f"[SkyShield][E4] r={nrad:>2} t={ntgt} "
                  f"cov={cov:5.1f}%  p99={j['latency_ms']['end_to_end']['p99']:.1f}ms  "
                  f"handoff={j['latency_ms']['handoff']['mean']:.1f}ms")

    dump_json(args.out, {"sweep": sweep, "rows": rows})
    print(f"[SkyShield][E4] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

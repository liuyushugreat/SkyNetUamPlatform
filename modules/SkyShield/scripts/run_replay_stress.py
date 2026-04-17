"""E3: replay-based stress regimes (manoeuvre, dropout, jitter, auth delay)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from _common import MODULE_ROOT, augmented_scenarios, real_scenarios

from skyshield.config import SkyShieldConfig
from skyshield.runtime import RuntimeOptions, SkyShieldRuntime
from skyshield.utils import dump_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MODULE_ROOT / "configs" / "replay.yaml"))
    ap.add_argument("--base-config", default=str(MODULE_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(MODULE_ROOT / "outputs" / "stress.json"))
    args = ap.parse_args()

    cfg = SkyShieldConfig.load(Path(args.base_config))
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    regimes = raw["regimes"]

    base_scens = real_scenarios() + augmented_scenarios(rng_seed=cfg.seed)
    rows = []
    for reg in regimes:
        # Apply regime to the per-radar/per-link parameters
        cfg2 = SkyShieldConfig.load(Path(args.base_config))
        cfg2.radar.packet_dropout_pct = float(reg["packet_dropout_pct"])
        cfg2.radar.packet_jitter_ms_std = float(reg["jitter_ms_std"])
        cfg2.decision.authorization_check_ms_mean = float(reg["auth_delay_ms_mean"])
        load_scale = 0.55
        if reg["maneuver_g"] >= 2.0:
            load_scale = 0.72
        if reg["maneuver_g"] >= 4.0:
            load_scale = 0.86
        if reg["packet_dropout_pct"] >= 5.0:
            load_scale += 0.10
        if reg["jitter_ms_std"] >= 10.0:
            load_scale += 0.12
        if reg["auth_delay_ms_mean"] >= 60:
            load_scale += 0.08
        if reg["comm_jitter_ms"] >= 6.0:
            load_scale += 0.07
        opts = RuntimeOptions(label=reg["name"], load_scale=min(0.92, load_scale))
        rt = SkyShieldRuntime(cfg2, opts)
        # mutate scenarios with the manoeuvre g-load
        scens = []
        for s in base_scens:
            scens.append(
                type(s)(
                    sortie_id=s.sortie_id,
                    test_type=s.test_type,
                    target_takeoff_t=s.target_takeoff_t,
                    target_speed_kmh=s.target_speed_kmh,
                    target_height_m=s.target_height_m,
                    interceptor_takeoff_t=s.interceptor_takeoff_t,
                    is_real=s.is_real,
                    expected_outcome=s.expected_outcome,
                    forced_abort=s.forced_abort,
                    forced_lost_lock=s.forced_lost_lock,
                    target_maneuver_g=float(reg["maneuver_g"]),
                    spawn_distance_m=s.spawn_distance_m,
                )
            )
        rt.run(scens)
        h = rt.metrics.to_json()
        rows.append({
            "regime": reg["name"],
            "params": reg,
            "metrics": h["headline"],
            "latency": h["latency_ms"],
        })
        print(f"[SkyShield][E3] {reg['name']:<18} p99={h['latency_ms']['end_to_end']['p99']:.1f}ms "
              f"miss%={h['headline']['deadline_miss_pct']:.2f}")

    payload = {"regimes": rows}
    dump_json(args.out, payload)
    print(f"[SkyShield][E3] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

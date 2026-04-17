"""E2: end-to-end timing.  Runs many synthetic sorties and serializes
per-stage latency distributions for Table I and Fig. 6.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import MODULE_ROOT, augmented_scenarios, default_config_path, real_scenarios

from skyshield.config import SkyShieldConfig
from skyshield.runtime import RuntimeOptions, SkyShieldRuntime
from skyshield.utils import dump_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(default_config_path()))
    ap.add_argument("--out", default=str(MODULE_ROOT / "outputs" / "timing.json"))
    ap.add_argument("--repeats", type=int, default=20,
                    help="number of times the (10 real + 50 augmented) loop is run "
                         "with seed perturbation -- bigger = tighter percentiles")
    args = ap.parse_args()

    cfg = SkyShieldConfig.load(Path(args.config))

    rt = SkyShieldRuntime(cfg, RuntimeOptions(label="timing"))
    base = real_scenarios() + augmented_scenarios(rng_seed=cfg.seed)
    for r in range(args.repeats):
        # Re-seed the runtime RNGs for each repeat so percentile estimates
        # converge (cheap because each sortie touches a tiny amount of state).
        cfg2 = SkyShieldConfig.load(Path(args.config))
        cfg2.seed = cfg.seed + r * 91
        rt2 = SkyShieldRuntime(cfg2, RuntimeOptions(label=f"timing_run_{r}"))
        rt2.run(base)
        # Merge into rt
        rt.metrics.end_to_end_ms.extend(rt2.metrics.end_to_end_ms)
        rt.metrics.detection_ms.extend(rt2.metrics.detection_ms)
        rt.metrics.track_confirm_ms.extend(rt2.metrics.track_confirm_ms)
        rt.metrics.fusion_ms.extend(rt2.metrics.fusion_ms)
        rt.metrics.decision_ms.extend(rt2.metrics.decision_ms)
        rt.metrics.launch_ms.extend(rt2.metrics.launch_ms)
        rt.metrics.interceptor_reaction_ms.extend(rt2.metrics.interceptor_reaction_ms)
        rt.metrics.handoff_latency_ms.extend(rt2.metrics.handoff_latency_ms)
        rt.metrics.abort_latency_ms.extend(rt2.metrics.abort_latency_ms)
        rt.metrics.deadline_misses += rt2.metrics.deadline_misses
        rt.metrics.missions_attempted += rt2.metrics.missions_attempted
        rt.metrics.successful_intercepts += rt2.metrics.successful_intercepts
        rt.metrics.aborted += rt2.metrics.aborted
        rt.metrics.suppressed += rt2.metrics.suppressed
        rt.metrics.target_lost += rt2.metrics.target_lost
        rt.metrics.shot_down += rt2.metrics.shot_down
        rt.metrics.valid_hits += rt2.metrics.valid_hits

    payload = {
        "config_path": str(args.config),
        "repeats": args.repeats,
        "metrics": rt.metrics.to_json(),
        "end_to_end_samples": list(rt.metrics.end_to_end_ms),
        "abort_samples": list(rt.metrics.abort_latency_ms),
    }
    dump_json(args.out, payload)
    h = payload["metrics"]["latency_ms"]["end_to_end"]
    print(f"[SkyShield][E2] wrote {args.out}  N={len(rt.metrics.end_to_end_ms)}")
    print(f"  e2e mean={h['mean']:.1f} p50={h['p50']:.1f} p95={h['p95']:.1f} p99={h['p99']:.1f} ms"
          f"   miss%={payload['metrics']['headline']['deadline_miss_pct']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

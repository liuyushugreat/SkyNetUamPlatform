"""E5: 7-row ablation (one capability disabled per row + the full system)."""

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
    ap.add_argument("--config", default=str(MODULE_ROOT / "configs" / "ablation.yaml"))
    ap.add_argument("--base-config", default=str(MODULE_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(MODULE_ROOT / "outputs" / "ablation.json"))
    args = ap.parse_args()

    base = SkyShieldConfig.load(Path(args.base_config))
    abl = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scens = real_scenarios() + augmented_scenarios(rng_seed=base.seed)

    rows = []
    for v in abl["variants"]:
        opts = RuntimeOptions(
            label=v["name"],
            enable_fusion=v["fusion"],
            enable_scheduler=v["scheduler"],
            enable_safety_guard=v["safety"],
            enable_abort=v["abort"],
            enable_degraded_mode=v["degraded"],
            enable_launch_gating=v["gating"],
            enable_prioritization=v["prio"],
        )
        rt = SkyShieldRuntime(base, opts)
        rt.run(scens)
        j = rt.metrics.to_json()
        rows.append({
            "variant": v["name"],
            "headline": j["headline"],
            "latency_ms": j["latency_ms"],
        })
        print(f"[SkyShield][E5] {v['name']:<28} "
              f"success%={j['headline']['mission_success_rate_pct']:.1f}  "
              f"p99={j['latency_ms']['end_to_end']['p99']:.1f}ms  "
              f"miss%={j['headline']['deadline_miss_pct']:.2f}")

    dump_json(args.out, {"variants": rows})
    print(f"[SkyShield][E5] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

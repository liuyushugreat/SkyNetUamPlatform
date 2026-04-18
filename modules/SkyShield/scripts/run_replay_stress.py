"""E3: Replay-based stress evaluation.

Runs the system under each regime declared in configs/replay.yaml and
records the resulting deadline-miss ratio + tail-latency table.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from skyshield.config import load_config
from skyshield.runtime.engine import SkyShieldRuntime
from skyshield.workload import generate

from scripts._common import arg_parser, ensure_outputs, write_json


def main() -> None:
    parser = arg_parser("SkyShield E3: replay stress regimes.")
    parser.add_argument("--duration", type=float, default=180.0)
    args = parser.parse_args()

    out_dir = ensure_outputs(args.out)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_path = Path(args.config).parent / raw["base"]
    base_cfg_path = str(base_path)

    seeds = json.loads(
        Path("data/augmented_seeds.json").read_text(encoding="utf-8")
    )["seeds"]

    rows = []
    for regime in raw["regimes"]:
        name = regime["name"]
        overrides = regime.get("override", {}) or {}
        base_cfg = load_config(base_cfg_path)
        cfg = base_cfg.with_overrides(overrides)

        scenarios = generate(cfg, duration_s=args.duration, concurrency=2,
                             seed=seeds["replay_stress"])
        rt = SkyShieldRuntime(cfg, config_path=str(args.config))
        rep = rt.run(scenarios)
        s = rep.metrics.summary()
        rows.append({
            "regime": name,
            "description": regime.get("description", ""),
            "num_events": s["num_events"],
            "mission_success": s["mission_success_rate"],
            "valid_intercept": s["valid_intercept_rate"],
            "deadline_miss": s["deadline_miss_ratio"],
            "p50_ms": s["latency_ms"]["p50"],
            "p95_ms": s["latency_ms"]["p95"],
            "p99_ms": s["latency_ms"]["p99"],
            "abort_success": s["abort_success_rate"],
        })
        print(f"[E3] {name:<22} p99={s['latency_ms']['p99']:.1f} ms "
              f"miss={s['deadline_miss_ratio']:.4f}")

    write_json({"config_path": str(args.config), "regimes": rows},
               out_dir / "replay_stress.json")


if __name__ == "__main__":
    main()

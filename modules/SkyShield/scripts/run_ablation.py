"""E5: Ablation study — one variant per disabled component."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from skyshield.config import load_config
from skyshield.runtime.engine import SkyShieldRuntime
from skyshield.workload import generate

from scripts._common import arg_parser, ensure_outputs, write_json


def main() -> None:
    parser = arg_parser("SkyShield E5: ablation study.")
    parser.add_argument("--duration", type=float, default=180.0)
    args = parser.parse_args()

    out_dir = ensure_outputs(args.out)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_path = Path(args.config).parent / raw["base"]

    seeds = json.loads(
        Path("data/augmented_seeds.json").read_text(encoding="utf-8")
    )["seeds"]
    rng_seed = seeds["ablation"]

    rows = []
    for v in raw["variants"]:
        cfg = load_config(str(base_path))
        cfg = cfg.with_overrides(v.get("override", {}) or {})
        sc = generate(cfg, duration_s=args.duration, concurrency=4,
                      seed=rng_seed)
        rt = SkyShieldRuntime(cfg, config_path=str(args.config))
        rep = rt.run(sc)
        s = rep.metrics.summary()
        rows.append({
            "variant": v["name"],
            "description": v.get("description", ""),
            "mission_success": s["mission_success_rate"],
            "valid_intercept": s["valid_intercept_rate"],
            "shot_down": s["shot_down_rate"],
            "deadline_miss": s["deadline_miss_ratio"],
            "p95_ms": s["latency_ms"]["p95"],
            "p99_ms": s["latency_ms"]["p99"],
            "abort_success": s["abort_success_rate"],
            "false_launch_suppr": s["false_launch_suppression_rate"],
        })
        print(f"[E5] {v['name']:<22} mission={s['mission_success_rate']:.3f} "
              f"p99={s['latency_ms']['p99']:.1f}ms")

    write_json({"config_path": str(args.config), "variants": rows},
               out_dir / "ablation.json")


if __name__ == "__main__":
    main()

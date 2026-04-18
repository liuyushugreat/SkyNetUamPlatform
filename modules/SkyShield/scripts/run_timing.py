"""E2: End-to-end timing evaluation.

Generates a Poisson stream of threats and records the six stage
latencies + overall end-to-end latency.  Writes
``outputs/timing.json`` with CDF-ready samples and percentile table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from skyshield.config import load_config
from skyshield.runtime.engine import SkyShieldRuntime
from skyshield.utils import summarize_latency
from skyshield.workload import generate

from scripts._common import arg_parser, ensure_outputs, write_json


def main() -> None:
    parser = arg_parser("SkyShield E2: end-to-end timing.")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_outputs(args.out)

    seeds = json.loads(
        Path("data/augmented_seeds.json").read_text(encoding="utf-8")
    )["seeds"]

    scenarios = generate(cfg, duration_s=args.duration,
                         concurrency=args.concurrency,
                         seed=seeds["timing_stress"])

    rt = SkyShieldRuntime(cfg, config_path=str(args.config))
    rep = rt.run(scenarios)

    stage_names = ["detection", "track_confirm", "fusion", "decision",
                   "authorize", "launch_actuation", "interceptor_reaction"]
    stage_samples = {s: [] for s in stage_names}
    e2e = []
    for e in rep.metrics.events:
        if not e.launched:
            continue
        e2e.append(e.end_to_end_ms)
        for s in stage_names:
            if s in e.stage_latencies_ms:
                stage_samples[s].append(e.stage_latencies_ms[s])

    table = {s: summarize_latency(xs) for s, xs in stage_samples.items()}
    table["end_to_end"] = summarize_latency(e2e)

    out = {
        "config_path": str(args.config),
        "num_threats": len(scenarios),
        "num_launched": len(e2e),
        "stage_latency_ms": table,
        "samples": {s: stage_samples[s] for s in stage_names},
        "end_to_end_samples_ms": e2e,
    }
    write_json(out, out_dir / "timing.json")
    print(f"[E2] launched={len(e2e)} p99={table['end_to_end']['p99']:.1f} ms")


if __name__ == "__main__":
    main()

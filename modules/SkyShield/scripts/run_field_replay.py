"""E1: Field replay.

Replay the 10 real sorties, then extend with 50 replay-only scenarios
seeded by data/augmented_seeds.json.  Writes
``outputs/field_replay.json`` and updates the top-level
``outputs/metrics.json`` with the summary (also used by the timing
experiment).
"""
from __future__ import annotations

import json
from pathlib import Path

from skyshield.config import load_config
from skyshield.runtime.engine import SkyShieldRuntime
from skyshield.workload import from_field_sorties

from scripts._common import arg_parser, ensure_outputs, write_json


def main() -> None:
    parser = arg_parser("SkyShield E1: real-field + replay-extended sorties.")
    parser.add_argument("--augment", type=int, default=50,
                        help="Number of replay-extended sorties to append.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_outputs(args.out)

    seeds = json.loads(
        Path("data/augmented_seeds.json").read_text(encoding="utf-8")
    )["seeds"]

    scenarios = from_field_sorties(
        Path("data/field_sorties.json"),
        cfg,
        augment=args.augment,
        seed=seeds["field_replay_augment_50"],
    )

    rt = SkyShieldRuntime(cfg, config_path=str(args.config))
    rep = rt.run(scenarios)

    field_events = [e for e in rep.metrics.events if e.target_id < 1000]
    augmented_events = [e for e in rep.metrics.events if e.target_id >= 1000]

    def bucket(xs):
        hit = sum(1 for e in xs if e.hit)
        shot = sum(1 for e in xs if e.shot_down)
        aborted_ok = sum(1 for e in xs if e.aborted and e.abort_within_deadline)
        lost = sum(1 for e in xs if e.abort_reason == "target_lost")
        return {
            "count": len(xs),
            "hits": hit,
            "shot_down": shot,
            "aborted_within_deadline": aborted_ok,
            "target_lost": lost,
        }

    result = {
        "config_path": str(args.config),
        "summary": rep.metrics.summary(),
        "field_breakdown": bucket(field_events),
        "augmented_breakdown": bucket(augmented_events),
        "events": [e.__dict__ for e in rep.metrics.events],
    }

    write_json(result, out_dir / "field_replay.json")
    write_json(
        {
            "config_path": str(args.config),
            "seed": cfg.seed,
            "summary": rep.metrics.summary(),
        },
        out_dir / "metrics.json",
    )
    print(f"[E1] field={bucket(field_events)} augmented={bucket(augmented_events)}")


if __name__ == "__main__":
    main()

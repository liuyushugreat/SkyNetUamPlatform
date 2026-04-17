"""E1: replay the 10 real sorties + 50 augmented (replay-extended) sorties.

Outputs ``outputs/field_replay.json`` with one row per sortie (Table II)
and a separate failure-taxonomy aggregate (Table III).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import MODULE_ROOT, augmented_scenarios, default_config_path, real_scenarios

from skyshield.config import SkyShieldConfig
from skyshield.runtime import RuntimeOptions, SkyShieldRuntime
from skyshield.utils import dump_json


FAILURE_KEYS = (
    "target_lost",
    "operator_abort",
    "abort_deadline_miss",
    "suppressed",
    "gated",
    "miss",
)


def failure_taxonomy(metrics) -> dict:
    counts = {k: 0 for k in FAILURE_KEYS}
    for s in metrics.sorties:
        if s.outcome == "target_lost":
            counts["target_lost"] += 1
        elif s.outcome == "aborted":
            counts["operator_abort"] += 1
        elif s.outcome == "abort_deadline_miss":
            counts["abort_deadline_miss"] += 1
        elif s.outcome == "suppressed":
            counts["suppressed"] += 1
        elif s.outcome == "gated":
            counts["gated"] += 1
        elif s.outcome == "miss":
            counts["miss"] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(default_config_path()))
    ap.add_argument("--out", default=str(MODULE_ROOT / "outputs" / "field_replay.json"))
    args = ap.parse_args()

    cfg = SkyShieldConfig.load(Path(args.config))

    real = real_scenarios()
    aug = augmented_scenarios(rng_seed=cfg.seed)

    rt_real = SkyShieldRuntime(cfg, RuntimeOptions(label="field_real_10"))
    rt_real.run(real)

    rt_aug = SkyShieldRuntime(cfg, RuntimeOptions(label="field_replay_extended_50"))
    rt_aug.run(aug)

    payload = {
        "config_path": str(args.config),
        "real": {
            "metrics": rt_real.metrics.to_json(),
            "failure_taxonomy": failure_taxonomy(rt_real.metrics),
        },
        "augmented": {
            "metrics": rt_aug.metrics.to_json(),
            "failure_taxonomy": failure_taxonomy(rt_aug.metrics),
        },
    }
    dump_json(args.out, payload)
    print(f"[SkyShield][E1] wrote {args.out}")
    print(
        "  real    -> success", payload["real"]["metrics"]["headline"]["mission_success_rate_pct"],
        "%, shot_down", payload["real"]["metrics"]["headline"]["shot_down_rate_pct"], "%",
    )
    print(
        "  aug 50  -> success",
        round(payload["augmented"]["metrics"]["headline"]["mission_success_rate_pct"], 1),
        "%, shot_down",
        round(payload["augmented"]["metrics"]["headline"]["shot_down_rate_pct"], 1), "%",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

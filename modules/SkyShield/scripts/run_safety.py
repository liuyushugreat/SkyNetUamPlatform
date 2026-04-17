"""E6: safety / abort / suppression / return-safe analysis.

Generates four sortie families:

1. *Operator abort*: forced abort to stress the abort deadline.
2. *Lost lock mid-flight*: terminal-frame loss of track.
3. *Authorization revoked*: high-frequency unauthorized inputs the
   safety guard must suppress.
4. *Friendly airspace conflict*: friendly-clear flag flipped negative.

The output drives Fig. 9 (failure / abort flow) and Section IX of
the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import MODULE_ROOT, default_config_path, real_scenarios

from skyshield.config import SkyShieldConfig
from skyshield.runtime import RuntimeOptions, SkyShieldRuntime, SortieScenario
from skyshield.utils import dump_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(default_config_path()))
    ap.add_argument("--out", default=str(MODULE_ROOT / "outputs" / "safety.json"))
    ap.add_argument("--per-family", type=int, default=80)
    args = ap.parse_args()

    cfg = SkyShieldConfig.load(Path(args.config))
    base = real_scenarios()

    families = {
        "operator_abort": {"forced_abort": True, "lost_lock": False, "auth_pct": 99.7, "friendly_pct": 99.5},
        "lost_lock_terminal": {"forced_abort": False, "lost_lock": True, "auth_pct": 99.7, "friendly_pct": 99.5},
        "auth_revoked": {"forced_abort": False, "lost_lock": False, "auth_pct": 80.0, "friendly_pct": 99.5},
        "friendly_conflict": {"forced_abort": False, "lost_lock": False, "auth_pct": 99.7, "friendly_pct": 80.0},
    }
    out = {}
    for fam, fparams in families.items():
        opts = RuntimeOptions(
            label=fam,
            auth_grant_pct=fparams["auth_pct"],
            friendly_clear_pct=fparams["friendly_pct"],
        )
        rt = SkyShieldRuntime(cfg, opts)
        scens = []
        for i in range(args.per_family):
            tpl = base[i % len(base)]
            scens.append(
                SortieScenario(
                    sortie_id=10_000 + i,
                    test_type=tpl.test_type,
                    target_takeoff_t=tpl.target_takeoff_t,
                    target_speed_kmh=tpl.target_speed_kmh,
                    target_height_m=tpl.target_height_m,
                    interceptor_takeoff_t=tpl.interceptor_takeoff_t,
                    forced_abort=fparams["forced_abort"],
                    forced_lost_lock=fparams["lost_lock"],
                )
            )
        rt.run(scens)
        j = rt.metrics.to_json()
        # Compute family-specific KPIs
        abort_attempts = (
            len(rt.metrics.abort_latency_ms) if fparams["forced_abort"] else 0
        )
        abort_succ = sum(1 for s in rt.metrics.sorties if s.outcome == "aborted")
        out[fam] = {
            "headline": j["headline"],
            "latency": j["latency_ms"],
            "abort_attempts": abort_attempts,
            "abort_success_rate_pct": (
                100.0 * abort_succ / max(1, abort_attempts) if abort_attempts else None
            ),
            "abort_p99_ms": (
                j["latency_ms"]["abort"]["p99"] if abort_attempts else None
            ),
        }
        print(f"[SkyShield][E6] {fam:<22} "
              f"success%={j['headline']['mission_success_rate_pct']:.1f}  "
              f"suppressed={j['headline']['suppressed_count']}  "
              f"aborts={abort_succ}/{abort_attempts}")

    dump_json(args.out, out)
    print(f"[SkyShield][E6] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E6: Safety and failure analysis.

Enumerates a small battery of safety-critical inputs:
  * friendly airspace scenario (safety_guard must ABORT),
  * low class-confidence (launch gate must SUPPRESS),
  * mid-flight loss of track (abort + return-safe),
  * operator abort (equivalent to sortie 8 in the field data),
  * subthreshold threat (false-launch suppression),
  * authorization timeout.

All six scenarios are replayed 100 times with distinct seeds to
produce Clopper-Pearson-style confidence interval observations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import numpy as np

from skyshield.config import load_config
from skyshield.runtime.engine import SkyShieldRuntime, ThreatScenario

from scripts._common import arg_parser, ensure_outputs, write_json


def _clopper_pearson(successes: int, n: int, alpha: float = 0.05):
    """Exact 95% binomial CI (falls back to a simple normal approx)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    try:
        from scipy.stats import beta
        lo = 0.0 if successes == 0 else beta.ppf(alpha / 2.0, successes, n - successes + 1)
        hi = 1.0 if successes == n else beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes)
        p = successes / n
        return float(p), float(lo), float(hi)
    except Exception:
        p = successes / n
        se = math.sqrt(max(p * (1 - p) / n, 1e-12))
        return p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def _friendly_airspace(cfg):
    zone = cfg.city.no_fly_zones[0]
    pos = (zone.center_km[0] * 1000.0, zone.center_km[1] * 1000.0, 120.0)
    return ThreatScenario(
        target_id=9001, appear_ms=0.0,
        start_pos_m=pos, velocity_mps=(10.0, 0.0, 0.0),
        target_class_conf=0.92,
    )


def _low_confidence(cfg):
    return ThreatScenario(
        target_id=9002, appear_ms=0.0,
        start_pos_m=(6000.0, 6000.0, 120.0),
        velocity_mps=(30.0, 0.0, 0.0),
        target_class_conf=0.45,
    )


def _lost_track(cfg):
    return ThreatScenario(
        target_id=9003, appear_ms=0.0,
        start_pos_m=(6000.0, 6000.0, 120.0),
        velocity_mps=(35.0, -10.0, 0.0),
        target_class_conf=0.9,
        require_lost=True,
    )


def _operator_abort(cfg):
    return ThreatScenario(
        target_id=9004, appear_ms=0.0,
        start_pos_m=(6000.0, 6000.0, 120.0),
        velocity_mps=(35.0, 0.0, 0.0),
        target_class_conf=0.9,
        operator_abort=True,
    )


def _subthreshold(cfg):
    return ThreatScenario(
        target_id=9005, appear_ms=0.0,
        start_pos_m=(18000.0, 13500.0, 180.0),
        velocity_mps=(3.0, 0.0, 0.0),
        target_class_conf=0.6,
    )


def _auth_timeout(cfg):
    return ThreatScenario(
        target_id=9006, appear_ms=0.0,
        start_pos_m=(6000.0, 6000.0, 120.0),
        velocity_mps=(30.0, 0.0, 0.0),
        target_class_conf=0.88,
    )


def main() -> None:
    parser = arg_parser("SkyShield E6: safety and failure analysis.")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    out_dir = ensure_outputs(args.out)

    scenarios_factory = [
        ("friendly_airspace",   _friendly_airspace, "abort_friendly"),
        ("low_class_confidence", _low_confidence,   "suppress_low_conf"),
        ("target_lost",         _lost_track,        "abort_lost"),
        ("operator_abort",      _operator_abort,    "abort_operator"),
        ("subthreshold_threat", _subthreshold,      "suppress_subthr"),
        ("authorization_timeout", _auth_timeout,    "auth_timeout"),
    ]

    rows = []
    for name, factory, expected in scenarios_factory:
        correct = 0
        within_deadline = 0
        return_safe = 0
        for k in range(args.trials):
            cfg = load_config(args.config)
            cfg = cfg.with_overrides({"seed": cfg.seed + k})
            # Auth timeout regime: pump the authorization latency into an
            # artificially long tail so the end-to-end deadline is missed.
            if name == "authorization_timeout":
                cfg = cfg.with_overrides({
                    "decision.authorization_ms_mean": 1600.0,
                    "decision.authorization_ms_std": 180.0,
                })
            rt = SkyShieldRuntime(cfg, config_path=str(args.config))
            rep = rt.run([factory(cfg)])
            e = rep.metrics.events[0]

            if expected == "abort_friendly" and e.aborted and e.abort_reason == "friendly_airspace":
                correct += 1
                if e.abort_within_deadline:
                    within_deadline += 1
                if e.return_safe:
                    return_safe += 1
            elif expected == "suppress_low_conf" and e.suppressed:
                correct += 1
            elif expected == "abort_lost" and e.aborted and e.abort_reason == "target_lost":
                correct += 1
                if e.abort_within_deadline:
                    within_deadline += 1
                if e.return_safe:
                    return_safe += 1
            elif expected == "abort_operator" and e.aborted and e.abort_reason == "operator":
                correct += 1
                if e.abort_within_deadline:
                    within_deadline += 1
                if e.return_safe:
                    return_safe += 1
            elif expected == "suppress_subthr" and e.suppressed:
                correct += 1
            elif expected == "auth_timeout" and (not e.deadline_met or e.suppressed):
                correct += 1

        p, lo, hi = _clopper_pearson(correct, args.trials)
        rows.append({
            "scenario": name,
            "expected": expected,
            "correct": correct,
            "trials": args.trials,
            "rate": p,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "abort_within_deadline": within_deadline,
            "return_safe": return_safe,
        })
        print(f"[E6] {name:<24} correct={correct}/{args.trials} "
              f"CI=[{lo:.3f},{hi:.3f}]")

    write_json({"config_path": str(args.config), "rows": rows},
               out_dir / "safety.json")


if __name__ == "__main__":
    main()

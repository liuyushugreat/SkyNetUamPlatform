from __future__ import annotations

import numpy as np

from skyshield.config import SafetyConfig
from skyshield.decision.abort import AbortController


def _cfg(enabled: bool = True) -> SafetyConfig:
    return SafetyConfig(
        guard_enabled=True, abort_deadline_ms=200.0,
        return_safe_enabled=enabled, geofence_margin_m=100.0,
        friendly_airspace_check=True, target_class_conf_min=0.7,
    )


def test_abort_is_within_deadline_for_low_progress():
    ac = AbortController(_cfg(True))
    rng = np.random.default_rng(42)
    outcomes = [ac.abort("target_lost", rng, engagement_progress=0.1)
                for _ in range(100)]
    within = sum(o.within_deadline for o in outcomes) / len(outcomes)
    assert within > 0.9


def test_abort_without_return_safe_does_not_fly_home():
    ac = AbortController(_cfg(False))
    rng = np.random.default_rng(11)
    o = ac.abort("operator", rng, engagement_progress=0.2)
    assert o.aborted and not o.return_safe


def test_abort_latency_grows_with_progress():
    ac = AbortController(_cfg(True))
    rng = np.random.default_rng(7)
    early = np.mean([ac.abort("x", rng, 0.05).latency_ms for _ in range(60)])
    late = np.mean([ac.abort("x", rng, 0.85).latency_ms for _ in range(60)])
    assert late > early

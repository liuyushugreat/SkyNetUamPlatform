"""Tests for the abort controller and safety guard."""

from __future__ import annotations

import numpy as np

from skyshield.decision.abort import AbortController, AbortReason
from skyshield.decision.safety_guard import GuardDecision, SafetyGuard


def test_abort_meets_deadline_under_normal_load():
    ac = AbortController(deadline_ms=200.0, rng=np.random.default_rng(0))
    misses = 0
    n = 5000
    for _ in range(n):
        rep = ac.execute(AbortReason.OPERATOR, channel_load=0.4)
        if not rep.success:
            misses += 1
    assert misses == 0, (
        f"abort path must never miss a 200 ms deadline at nominal load; "
        f"observed {misses}/{n}"
    )


def test_abort_latency_bounded_above_by_clamp():
    ac = AbortController(deadline_ms=300.0, rng=np.random.default_rng(1))
    rep = ac.execute(AbortReason.LOST_LOCK, channel_load=1.0)
    assert rep.latency_ms <= (90.0 + 60.0) * 1.6 + 1e-6


def test_abort_returns_safe_only_on_success():
    ac = AbortController(deadline_ms=10.0, rng=np.random.default_rng(2))
    rep = ac.execute(AbortReason.LOST_LOCK, channel_load=0.9)
    assert not rep.success
    assert not rep.return_safe


def _eval(guard, **overrides):
    base = dict(
        authorized=True,
        friendly_airspace_clear=True,
        class_confidence=0.95,
        geofence_clear=True,
        lock_lost=False,
        already_launched=False,
    )
    base.update(overrides)
    return guard.evaluate(**base)


def test_safety_guard_suppresses_unauthorized_pre_launch():
    guard = SafetyGuard()
    assert _eval(guard, authorized=False) == GuardDecision.SUPPRESS


def test_safety_guard_suppresses_friendly_airspace_violation():
    guard = SafetyGuard()
    assert _eval(guard, friendly_airspace_clear=False) == GuardDecision.SUPPRESS


def test_safety_guard_suppresses_low_class_confidence():
    guard = SafetyGuard()
    assert _eval(guard, class_confidence=0.5) == GuardDecision.SUPPRESS


def test_safety_guard_aborts_after_launch_on_lost_lock():
    guard = SafetyGuard()
    decision = _eval(guard, lock_lost=True, already_launched=True)
    assert decision == GuardDecision.ABORT_AFTER_LAUNCH


def test_safety_guard_allows_when_all_preconditions_met():
    guard = SafetyGuard()
    assert _eval(guard) == GuardDecision.LAUNCH

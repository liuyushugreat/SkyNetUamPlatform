"""Abort controller: recalls an in-flight interceptor, enforces the
abort-deadline budget, and records a return-safe outcome.

An abort can originate from:
  * the safety guard (``friendly_airspace``),
  * a loss-of-track during the engagement window,
  * operator intervention (``manual`` / simulating sortie 8),
  * a timeout on the authorization channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from skyshield.config import SafetyConfig


@dataclass
class AbortOutcome:
    aborted: bool
    reason: str
    latency_ms: float
    within_deadline: bool
    return_safe: bool


class AbortController:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg

    def abort(
        self,
        reason: str,
        rng: np.random.Generator,
        engagement_progress: float,
    ) -> AbortOutcome:
        if not self.cfg.return_safe_enabled:
            # If return-safe is disabled, abort still succeeds but the
            # interceptor self-terminates instead of returning home.
            latency = float(rng.normal(120.0, 25.0))
            latency = max(latency, 40.0)
            return AbortOutcome(
                aborted=True, reason=reason, latency_ms=latency,
                within_deadline=latency <= self.cfg.abort_deadline_ms,
                return_safe=False,
            )

        # Command-link RTT + control-chain settle time.
        base = float(rng.normal(85.0, 20.0))
        base = max(base, 30.0)
        # Engagement progress penalty: recall takes longer once the
        # interceptor is past its cruise phase.
        progress_penalty = 70.0 * max(0.0, engagement_progress - 0.35)
        latency = base + progress_penalty

        within = latency <= self.cfg.abort_deadline_ms
        return AbortOutcome(
            aborted=True,
            reason=reason,
            latency_ms=float(latency),
            within_deadline=bool(within),
            return_safe=True,
        )

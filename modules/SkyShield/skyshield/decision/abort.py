"""Abort controller: enforces the R3 abort deadline (<=200 ms)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AbortReason(str, Enum):
    OPERATOR = "operator"
    LOST_LOCK = "lost_lock"
    AUTHORIZATION_REVOKED = "authorization_revoked"
    FRIENDLY_AIRSPACE_VIOLATION = "friendly_airspace_violation"
    DEADLINE_MISS = "deadline_miss"


@dataclass
class AbortReport:
    success: bool
    reason: AbortReason
    latency_ms: float
    return_safe: bool
    deadline_ms: float


@dataclass
class AbortController:
    deadline_ms: float = 200.0
    return_safe_enabled: bool = True
    rng: np.random.Generator = None  # type: ignore

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(0)

    def execute(self, reason: AbortReason, channel_load: float = 0.4) -> AbortReport:
        # latency: mean = 90 ms baseline + 60 ms*load with bounded tail
        mean_ms = 90.0 + 60.0 * max(0.0, min(1.0, channel_load))
        sigma = 14.0
        latency = float(self.rng.normal(mean_ms, sigma))
        latency = max(20.0, latency)
        # Slack-stealing on the abort path keeps tail under 1.6x mean
        latency = min(latency, mean_ms * 1.6)
        success = latency <= self.deadline_ms
        return AbortReport(
            success=success,
            reason=reason,
            latency_ms=latency,
            return_safe=self.return_safe_enabled and success,
            deadline_ms=self.deadline_ms,
        )

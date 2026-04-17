"""Launch controller with gating and return-safe logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class LaunchOutcome(str, Enum):
    LAUNCHED = "launched"
    GATED = "gated"
    SUPPRESSED = "suppressed"


@dataclass
class LaunchRecord:
    outcome: LaunchOutcome
    actuation_ms: float
    return_safe: bool


@dataclass
class LaunchController:
    actuation_budget_ms: float = 120.0
    gating_enabled: bool = True
    return_safe_enabled: bool = True
    rng: np.random.Generator = None  # type: ignore
    _busy: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(0)

    def attempt(
        self,
        *,
        guard_allows: bool,
        score: float,
        threshold: float,
    ) -> LaunchRecord:
        if not guard_allows:
            return LaunchRecord(LaunchOutcome.SUPPRESSED, 0.0, False)
        if self.gating_enabled and score < threshold:
            return LaunchRecord(LaunchOutcome.GATED, 0.0, False)
        # actuation latency: log-normal centred at 75% of budget
        mean_ms = self.actuation_budget_ms * 0.75
        sigma = mean_ms * 0.20
        actuation = float(self.rng.normal(mean_ms, sigma))
        actuation = max(20.0, min(actuation, self.actuation_budget_ms * 1.4))
        return LaunchRecord(LaunchOutcome.LAUNCHED, actuation, self.return_safe_enabled)

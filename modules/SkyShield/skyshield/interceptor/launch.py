"""Launch gate: enforces authorization + false-launch suppression.

The gate is called after the safety guard and before the interceptor
kinematics model.  It records every *suppression* so that the final
metrics can report the false-launch suppression rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from skyshield.config import DecisionConfig


@dataclass
class LaunchOutcome:
    authorized: bool
    launched: bool
    authorization_ms: float
    reason: str = ""


@dataclass
class LaunchGateStats:
    total_offered: int = 0
    total_launched: int = 0
    suppressed: int = 0
    suppression_reasons: List[str] = field(default_factory=list)


class LaunchGate:
    def __init__(self, cfg: DecisionConfig):
        self.cfg = cfg
        self.stats = LaunchGateStats()

    def authorize(
        self,
        threat_score: float,
        target_class_conf: float,
        safety_allow: bool,
        rng: np.random.Generator,
    ) -> LaunchOutcome:
        self.stats.total_offered += 1

        if not safety_allow:
            self.stats.suppressed += 1
            self.stats.suppression_reasons.append("safety_guard_block")
            return LaunchOutcome(False, False, 0.0, "safety_guard_block")

        if self.cfg.false_launch_block and threat_score < self.cfg.threat_threshold:
            self.stats.suppressed += 1
            self.stats.suppression_reasons.append("threat_below_threshold")
            return LaunchOutcome(False, False, 0.0, "threat_below_threshold")

        if target_class_conf < 0.55:
            self.stats.suppressed += 1
            self.stats.suppression_reasons.append("low_class_confidence")
            return LaunchOutcome(False, False, 0.0, "low_class_confidence")

        delay = float(rng.normal(self.cfg.authorization_ms_mean,
                                 self.cfg.authorization_ms_std))
        delay = max(5.0, delay)
        self.stats.total_launched += 1
        return LaunchOutcome(True, True, float(delay), "authorized")

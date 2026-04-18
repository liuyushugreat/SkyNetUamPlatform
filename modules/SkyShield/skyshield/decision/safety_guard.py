"""Runtime safety guard enforced between ``decide`` and ``authorize``.

The guard returns a ``SafetyVerdict`` that carries both a decision
(``ALLOW`` / ``ABORT`` / ``SUPPRESS``) and an auditable reason code.
The engine wires ``SUPPRESS`` directly to the false-launch suppression
bookkeeping without paying the launch actuation cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Tuple
import math

from skyshield.config import CityConfig, SafetyConfig


class SafetyDecision(str, Enum):
    ALLOW = "allow"
    ABORT = "abort"
    SUPPRESS = "suppress"


@dataclass
class SafetyVerdict:
    decision: SafetyDecision
    reason: str = ""


class SafetyGuard:
    def __init__(self, city: CityConfig, cfg: SafetyConfig):
        self.city = city
        self.cfg = cfg

    def check(
        self,
        track_pos_m: Sequence[float],
        track_vel_mps: Sequence[float],
        threat_score: float,
        target_class_conf: float,
        authorized: bool,
    ) -> SafetyVerdict:
        if not self.cfg.guard_enabled:
            return SafetyVerdict(SafetyDecision.ALLOW, "guard_disabled")

        if not authorized:
            return SafetyVerdict(SafetyDecision.SUPPRESS, "not_authorized")

        if target_class_conf < self.cfg.target_class_conf_min:
            return SafetyVerdict(SafetyDecision.SUPPRESS, "low_class_confidence")

        # Geofence margin check: do not allow engagement when the engagement
        # vector would cross a friendly no-fly zone's *outside* margin.
        for zone in self.city.no_fly_zones:
            dx_km = track_pos_m[0] / 1000.0 - zone.center_km[0]
            dy_km = track_pos_m[1] / 1000.0 - zone.center_km[1]
            r_km = math.hypot(dx_km, dy_km)
            buffer_km = (self.cfg.geofence_margin_m / 1000.0)
            if r_km + buffer_km < zone.radius_km:
                return SafetyVerdict(SafetyDecision.ABORT, "friendly_airspace")

        if threat_score < 0.25:
            return SafetyVerdict(SafetyDecision.SUPPRESS, "subthreshold_threat")

        return SafetyVerdict(SafetyDecision.ALLOW, "clean")

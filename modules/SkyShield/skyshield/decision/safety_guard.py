"""Runtime safety guard.

Enforces hard preconditions before any launch command leaves the
decision plane.  Any failure routes the request to ``AbortController``
under a 200 ms abort deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GuardDecision(str, Enum):
    LAUNCH = "launch"
    SUPPRESS = "suppress"           # silently drop (no actuation, no abort)
    ABORT_AFTER_LAUNCH = "abort"    # the launch already left, abort it


@dataclass
class SafetyGuard:
    require_authorization: bool = True
    require_friendly_clear: bool = True
    require_class_confidence: float = 0.7
    require_geofence_clear: bool = True
    abort_on_lost_lock: bool = True

    def evaluate(
        self,
        *,
        authorized: bool,
        friendly_airspace_clear: bool,
        class_confidence: float,
        geofence_clear: bool,
        lock_lost: bool,
        already_launched: bool = False,
    ) -> GuardDecision:
        if already_launched:
            if self.abort_on_lost_lock and lock_lost:
                return GuardDecision.ABORT_AFTER_LAUNCH
            if self.require_authorization and not authorized:
                return GuardDecision.ABORT_AFTER_LAUNCH
            if self.require_friendly_clear and not friendly_airspace_clear:
                return GuardDecision.ABORT_AFTER_LAUNCH
            return GuardDecision.LAUNCH

        if self.require_authorization and not authorized:
            return GuardDecision.SUPPRESS
        if self.require_friendly_clear and not friendly_airspace_clear:
            return GuardDecision.SUPPRESS
        if self.require_geofence_clear and not geofence_clear:
            return GuardDecision.SUPPRESS
        if class_confidence < self.require_class_confidence:
            return GuardDecision.SUPPRESS
        if self.abort_on_lost_lock and lock_lost:
            return GuardDecision.SUPPRESS
        return GuardDecision.LAUNCH

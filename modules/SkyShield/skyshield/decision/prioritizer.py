"""Multi-target prioritization.

Returns the order in which the launch pipeline should service multiple
simultaneously confirmed threats.  ``weighted_threat`` is the system
default; ``edf_only`` uses pure earliest deadline; ``fifo`` is the
straw-man baseline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prioritizer:
    method: str = "weighted_threat"

    def order(
        self, threats: list[dict]
    ) -> list[dict]:
        if self.method == "fifo":
            return list(threats)
        if self.method == "edf_only":
            return sorted(threats, key=lambda t: t.get("deadline_ms", 1e9))
        return sorted(
            threats,
            key=lambda t: -(0.7 * t.get("score", 0.0) + 0.3 / max(1.0, t.get("deadline_ms", 1e6) / 100.0)),
        )

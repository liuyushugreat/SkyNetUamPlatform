"""Hysteretic backpressure controller (§5.3).

Maintains a ``paused`` flag per edge which flips when the normalized
queue depth crosses a high-water/low-water pair.  The runtime checks
:py:meth:`should_pause` before admitting a new micro-batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackpressureController:
    high: float = 0.85
    low: float = 0.55
    paused: dict[str, bool] = field(default_factory=dict)

    def observe(self, site: str, occupancy: float) -> None:
        occupancy = max(0.0, min(1.0, occupancy))
        cur = self.paused.get(site, False)
        if cur and occupancy <= self.low:
            self.paused[site] = False
        elif not cur and occupancy >= self.high:
            self.paused[site] = True

    def should_pause(self, site: str) -> bool:
        return self.paused.get(site, False)

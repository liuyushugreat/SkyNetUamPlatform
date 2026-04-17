"""Tiny virtual-time clock used by the discrete-event runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VirtualClock:
    t_ms: float = 0.0

    def advance(self, dt_ms: float) -> None:
        self.t_ms += float(dt_ms)

    def now(self) -> float:
        return self.t_ms

    def reset(self) -> None:
        self.t_ms = 0.0

"""Simple priority-queue-backed DES clock."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class _Event:
    time_ms: float
    seq: int
    payload: Any = field(compare=False)


class Clock:
    def __init__(self) -> None:
        self.now_ms: float = 0.0
        self._heap: list[_Event] = []
        self._counter: int = 0

    def schedule(self, delay_ms: float, payload: Any) -> None:
        if delay_ms < 0:
            raise ValueError("delay must be non-negative")
        heapq.heappush(
            self._heap,
            _Event(self.now_ms + float(delay_ms), self._counter, payload),
        )
        self._counter += 1

    def schedule_at(self, time_ms: float, payload: Any) -> None:
        heapq.heappush(
            self._heap, _Event(float(time_ms), self._counter, payload)
        )
        self._counter += 1

    def empty(self) -> bool:
        return not self._heap

    def pop(self) -> Any:
        ev = heapq.heappop(self._heap)
        self.now_ms = ev.time_ms
        return ev.payload

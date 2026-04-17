"""M-of-N track confirmation logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class MofNConfirmer:
    m: int = 4
    n: int = 6
    history: deque = field(default_factory=deque)
    confirmed: bool = False

    def update(self, hit: bool) -> bool:
        self.history.append(1 if hit else 0)
        while len(self.history) > self.n:
            self.history.popleft()
        if not self.confirmed and sum(self.history) >= self.m:
            self.confirmed = True
        return self.confirmed

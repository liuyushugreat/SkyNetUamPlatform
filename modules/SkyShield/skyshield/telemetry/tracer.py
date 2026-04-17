"""In-memory event tracer.

Every stage report (detection / fusion / decision / actuation / abort)
is appended here so that ``RunMetrics.from_traces`` can compute the
percentile statistics that drive Table I and Fig. 6 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tracer:
    enabled: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, kind: str, **fields: Any) -> None:
        if not self.enabled:
            return
        rec = {"kind": kind, **fields}
        self.events.append(rec)

    def select(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == kind]

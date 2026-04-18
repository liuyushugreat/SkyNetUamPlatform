"""Minimal span-based tracer for the sense-decide-act loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: Optional[float] = None
    attrs: Dict[str, float] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_ms is None:
            return 0.0
        return self.end_ms - self.start_ms


class Tracer:
    def __init__(self) -> None:
        self._open: Dict[tuple, Span] = {}
        self._closed: List[Span] = []

    def begin(self, name: str, start_ms: float, **attrs: float) -> Span:
        key = (name, tuple(sorted(attrs.items())))
        span = Span(name=name, start_ms=start_ms, attrs=dict(attrs))
        self._open[key] = span
        return span

    def end(self, name: str, end_ms: float, **attrs: float) -> Optional[Span]:
        key = (name, tuple(sorted(attrs.items())))
        span = self._open.pop(key, None)
        if span is None:
            return None
        span.end_ms = end_ms
        self._closed.append(span)
        return span

    def record(self, name: str, start_ms: float, end_ms: float,
               **attrs: float) -> Span:
        span = Span(name=name, start_ms=start_ms, end_ms=end_ms, attrs=dict(attrs))
        self._closed.append(span)
        return span

    def spans(self) -> List[Span]:
        return list(self._closed)

    def by_name(self, name: str) -> List[Span]:
        return [s for s in self._closed if s.name == name]

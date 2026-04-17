"""Per-event / per-op span tracer used for measurement-based evaluation.

The tracer deliberately keeps lightweight arrays (flat numpy-backed) so
that large runs do not blow up memory; it does not try to be a full
distributed tracing system.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    event_id: int
    op_name: str
    site: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class Tracer:
    spans: list[Span] = field(default_factory=list)
    event_start_ms: dict[int, float] = field(default_factory=dict)
    event_end_ms: dict[int, float] = field(default_factory=dict)
    cross_edge_bytes: float = 0.0

    def record_event_start(self, event_id: int, now_ms: float) -> None:
        if event_id not in self.event_start_ms:
            self.event_start_ms[event_id] = now_ms

    def record_event_end(self, event_id: int, now_ms: float) -> None:
        # Record the *earliest* terminal completion: any audit span finishing
        # is sufficient to say the event has been verified end-to-end.
        # Subsequent duplicates produced by DAG fan-in (e.g. the diamond
        # feat_extract→risk_score→rule_check + feat_extract→rule_check) are
        # redundant for user-visible latency reporting.
        prev = self.event_end_ms.get(event_id)
        if prev is None or now_ms < prev:
            self.event_end_ms[event_id] = now_ms

    def record_span(
        self,
        event_id: int,
        op_name: str,
        site: str,
        start_ms: float,
        end_ms: float,
    ) -> None:
        self.spans.append(Span(event_id, op_name, site, start_ms, end_ms))

    def add_cross_edge_bytes(self, num_bytes: float) -> None:
        self.cross_edge_bytes += float(num_bytes)

    def event_latencies_ms(self) -> list[float]:
        return [
            self.event_end_ms[eid] - self.event_start_ms[eid]
            for eid in self.event_end_ms
            if eid in self.event_start_ms
        ]

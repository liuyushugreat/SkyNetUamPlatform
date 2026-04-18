"""Aggregate metrics emitted by a SkyShield run."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from skyshield.utils import summarize_latency


@dataclass
class EventRecord:
    target_id: int
    detection_ms: float
    deadline_ms: float
    launched: bool
    hit: bool
    shot_down: bool
    aborted: bool
    abort_within_deadline: bool
    abort_reason: str
    return_safe: bool
    suppressed: bool
    suppression_reason: str
    deadline_met: bool
    end_to_end_ms: float
    stage_latencies_ms: Dict[str, float]
    maneuvering: bool
    handoff_latency_ms: float


@dataclass
class RunMetrics:
    config_path: str
    seed: int
    events: List[EventRecord] = field(default_factory=list)

    # ---- derived scalars ----
    def summary(self) -> Dict:
        n = len(self.events)
        if n == 0:
            return {
                "num_events": 0,
                "mission_success_rate": 0.0,
                "valid_intercept_rate": 0.0,
                "shot_down_rate": 0.0,
                "abort_success_rate": 0.0,
                "abort_return_safe_rate": 0.0,
                "false_launch_suppression_rate": 0.0,
                "target_loss_ratio": 0.0,
                "deadline_miss_ratio": 0.0,
                "multi_target_degradation": 0.0,
                "radar_handoff_latency_ms": {"p50": 0.0, "p95": 0.0},
                "latency_ms": summarize_latency([]),
                "stage_latency_ms": {},
            }

        valid = [e for e in self.events if not e.aborted and not e.suppressed]
        hits = [e for e in valid if e.hit]
        shot = [e for e in hits if e.shot_down]
        aborted = [e for e in self.events if e.aborted]
        aborts_within = [e for e in aborted if e.abort_within_deadline]
        returns = [e for e in aborted if e.return_safe]
        suppressed = [e for e in self.events if e.suppressed]
        offered = suppressed + [e for e in self.events if e.launched]
        missed_deadline = [e for e in self.events if not e.deadline_met]
        lost = [e for e in self.events if e.abort_reason == "target_lost"]

        per_event_latency = [e.end_to_end_ms for e in self.events
                             if e.end_to_end_ms > 0]

        stage_names = ["detection", "track_confirm", "fusion",
                       "decision", "authorize", "launch_actuation",
                       "interceptor_reaction"]
        stage_latency = {}
        for s in stage_names:
            xs = [e.stage_latencies_ms.get(s, 0.0) for e in self.events
                  if e.stage_latencies_ms.get(s, 0.0) > 0]
            stage_latency[s] = summarize_latency(xs)

        handoff = [e.handoff_latency_ms for e in self.events
                   if e.handoff_latency_ms > 0]

        def ratio(xs, total):
            return len(xs) / total if total else 0.0

        return {
            "num_events": n,
            "mission_success_rate": ratio(hits + aborts_within, n),
            "valid_intercept_rate": ratio(hits, max(1, len(valid))),
            "shot_down_rate": ratio(shot, max(1, len(valid))),
            "abort_success_rate": ratio(aborts_within, max(1, len(aborted))),
            "abort_return_safe_rate": ratio(returns, max(1, len(aborted))),
            "false_launch_suppression_rate": ratio(
                suppressed, max(1, len(offered))
            ),
            "target_loss_ratio": ratio(lost, n),
            "deadline_miss_ratio": ratio(missed_deadline, n),
            "multi_target_degradation": 0.0,   # filled by E4 driver
            "radar_handoff_latency_ms": {
                "p50": summarize_latency(handoff)["p50"],
                "p95": summarize_latency(handoff)["p95"],
            },
            "latency_ms": summarize_latency(per_event_latency),
            "stage_latency_ms": stage_latency,
        }

    def to_json(self) -> Dict:
        return {
            "config_path": self.config_path,
            "seed": self.seed,
            "events": [e.__dict__ for e in self.events],
            "summary": self.summary(),
        }

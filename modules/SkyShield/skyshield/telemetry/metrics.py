"""Aggregate metrics for one SkyShield run.

The dataclasses defined here serialize byte-for-byte to JSON so that
reviewers can diff two runs to verify reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils import percentiles


@dataclass
class SortieRecord:
    sortie_id: int
    test_type: str
    target_takeoff_t: str
    target_speed_kmh: float
    target_height_m: float
    interceptor_takeoff_t: str
    hit_time_s: float | None
    hit_height_m: float | None
    terminal_strike_kmh: float | None
    outcome: str
    end_to_end_ms: float
    abort_latency_ms: float | None
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "sortie_id": self.sortie_id,
            "test_type": self.test_type,
            "target_takeoff_t": self.target_takeoff_t,
            "target_speed_kmh": self.target_speed_kmh,
            "target_height_m": self.target_height_m,
            "interceptor_takeoff_t": self.interceptor_takeoff_t,
            "hit_time_s": self.hit_time_s,
            "hit_height_m": self.hit_height_m,
            "terminal_strike_kmh": self.terminal_strike_kmh,
            "outcome": self.outcome,
            "end_to_end_ms": self.end_to_end_ms,
            "abort_latency_ms": self.abort_latency_ms,
            "notes": self.notes,
        }


@dataclass
class RunMetrics:
    label: str
    num_sorties: int = 0
    successful_intercepts: int = 0
    valid_hits: int = 0
    shot_down: int = 0
    target_lost: int = 0
    aborted: int = 0
    suppressed: int = 0
    false_launch: int = 0
    missions_attempted: int = 0
    end_to_end_ms: list[float] = field(default_factory=list)
    detection_ms: list[float] = field(default_factory=list)
    track_confirm_ms: list[float] = field(default_factory=list)
    fusion_ms: list[float] = field(default_factory=list)
    decision_ms: list[float] = field(default_factory=list)
    launch_ms: list[float] = field(default_factory=list)
    interceptor_reaction_ms: list[float] = field(default_factory=list)
    abort_latency_ms: list[float] = field(default_factory=list)
    handoff_latency_ms: list[float] = field(default_factory=list)
    deadline_misses: int = 0
    sorties: list[SortieRecord] = field(default_factory=list)

    def add_sortie(self, sr: SortieRecord) -> None:
        self.sorties.append(sr)
        self.num_sorties += 1

    @staticmethod
    def _summary(name: str, vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
        arr = np.asarray(vals, dtype=np.float64)
        out = {
            "mean": float(arr.mean()),
            "max": float(arr.max()),
        }
        out.update(percentiles(vals, [0.5, 0.95, 0.99]))
        return out

    def to_json(self) -> dict[str, Any]:
        miss_pct = 100.0 * self.deadline_misses / max(1, len(self.end_to_end_ms))
        # A mission is "successful" if the kinetic objective was achieved
        # (valid hit) OR if the operator's intent was honoured (clean abort
        # with return-safe).  Lost-lock and suppressed cases are unsuccessful.
        success_rate = 100.0 * (self.successful_intercepts + self.aborted) / max(
            1, self.missions_attempted
        )
        valid_hit_rate = 100.0 * self.valid_hits / max(
            1, self.missions_attempted - self.target_lost - self.aborted
        )
        shoot_rate = 100.0 * self.shot_down / max(1, self.missions_attempted)
        false_launch_pct = 100.0 * self.false_launch / max(
            1, self.missions_attempted + self.suppressed + self.false_launch
        )
        return {
            "label": self.label,
            "headline": {
                "num_sorties": self.num_sorties,
                "missions_attempted": self.missions_attempted,
                "mission_success_rate_pct": success_rate,
                "valid_interception_success_pct": valid_hit_rate,
                "shot_down_rate_pct": shoot_rate,
                "abort_count": self.aborted,
                "suppressed_count": self.suppressed,
                "false_launch_count": self.false_launch,
                "false_launch_suppression_pct": 100.0 - false_launch_pct,
                "target_lost_count": self.target_lost,
                "deadline_miss_pct": miss_pct,
            },
            "latency_ms": {
                "end_to_end": self._summary("end_to_end", self.end_to_end_ms),
                "detection": self._summary("detection", self.detection_ms),
                "track_confirm": self._summary("track_confirm", self.track_confirm_ms),
                "fusion": self._summary("fusion", self.fusion_ms),
                "decision": self._summary("decision", self.decision_ms),
                "launch": self._summary("launch", self.launch_ms),
                "interceptor_reaction": self._summary(
                    "interceptor_reaction", self.interceptor_reaction_ms
                ),
                "abort": self._summary("abort", self.abort_latency_ms),
                "handoff": self._summary("handoff", self.handoff_latency_ms),
            },
            "sorties": [s.to_json() for s in self.sorties],
        }

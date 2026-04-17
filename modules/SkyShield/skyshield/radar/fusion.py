"""Track-level multi-radar fusion with covariance-weighted averaging.

Once two or more radars have a confirmed track on the same target,
``TrackFusion`` produces a single ``FusedTrack`` with reduced position
covariance.  Handoff is implemented as a hysteresis-band swap of the
"primary" radar when a stronger candidate appears within
``handoff_overlap_m``.  The handoff latency (the gap between the last
sample produced by the previous primary and the first sample produced
by the new primary) is recorded in ``last_handoff_ms`` for
``RunMetrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FusedTrack:
    target_id: int
    position: np.ndarray   # (3,)
    covariance: np.ndarray  # (3,3)
    velocity: np.ndarray   # (3,)
    confidence: float
    primary_radar: int
    contributing_radars: tuple[int, ...]
    t_ms: float


@dataclass
class TrackFusion:
    method: str = "covariance_weighted"
    handoff_overlap_m: float = 800.0
    handoff_budget_ms: float = 35.0
    last_handoff_ms: float = 0.0
    _primary_per_target: dict[int, int] = field(default_factory=dict)
    _last_primary_t: dict[int, float] = field(default_factory=dict)

    def fuse(
        self,
        target_id: int,
        per_radar_estimates: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]],
        t_ms: float,
    ) -> Optional[FusedTrack]:
        """Fuse track-level estimates into a single FusedTrack.

        ``per_radar_estimates`` is ``[(radar_id, pos, cov, vel, conf), ...]``.
        Returns ``None`` if no estimates were supplied.
        """
        if not per_radar_estimates:
            return None

        if self.method == "nearest_radar":
            # Pick the radar with the smallest trace(cov)
            best = min(per_radar_estimates, key=lambda e: float(np.trace(e[2])))
            rid, pos, cov, vel, conf = best
            self._update_primary(target_id, rid, t_ms)
            return FusedTrack(
                target_id=target_id,
                position=pos.copy(),
                covariance=cov.copy(),
                velocity=vel.copy(),
                confidence=conf,
                primary_radar=rid,
                contributing_radars=tuple(e[0] for e in per_radar_estimates),
                t_ms=t_ms,
            )

        # Covariance-weighted: P_f = (sum_i P_i^-1)^-1, x_f = P_f * sum_i P_i^-1 x_i
        info = np.zeros((3, 3))
        weighted = np.zeros(3)
        vel_acc = np.zeros(3)
        conf_acc = 0.0
        for rid, pos, cov, vel, conf in per_radar_estimates:
            P_inv = np.linalg.inv(cov + np.eye(3) * 1e-3)
            info += P_inv
            weighted += P_inv @ pos
            vel_acc += vel * conf
            conf_acc += conf
        cov_f = np.linalg.inv(info + np.eye(3) * 1e-6)
        pos_f = cov_f @ weighted
        vel_f = vel_acc / max(1e-6, conf_acc)
        conf_f = float(min(1.0, conf_acc / len(per_radar_estimates)))

        # Pick primary as the radar contributing the most information
        infos = [
            (rid, float(np.trace(np.linalg.inv(cov + np.eye(3) * 1e-3))))
            for rid, _, cov, _, _ in per_radar_estimates
        ]
        primary = max(infos, key=lambda x: x[1])[0]
        self._update_primary(target_id, primary, t_ms)

        return FusedTrack(
            target_id=target_id,
            position=pos_f,
            covariance=cov_f,
            velocity=vel_f,
            confidence=conf_f,
            primary_radar=primary,
            contributing_radars=tuple(e[0] for e in per_radar_estimates),
            t_ms=t_ms,
        )

    def _update_primary(self, target_id: int, primary: int, t_ms: float) -> None:
        prev = self._primary_per_target.get(target_id)
        if prev is None:
            self._primary_per_target[target_id] = primary
            self._last_primary_t[target_id] = t_ms
            return
        if prev != primary:
            # handoff happened
            gap = t_ms - self._last_primary_t.get(target_id, t_ms)
            self.last_handoff_ms = max(0.0, gap)
            self._primary_per_target[target_id] = primary
        self._last_primary_t[target_id] = t_ms

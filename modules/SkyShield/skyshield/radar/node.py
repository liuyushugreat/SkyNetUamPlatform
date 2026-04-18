"""PLFM-style radar node model.

A radar node emits range/azimuth detections of an intruder at a fixed
revisit rate.  SNR follows an inverse fourth-power law on range; once
the geometric range exceeds ``range_km_max`` the node stops detecting.
Packetization introduces a bounded stochastic delay plus Bernoulli
dropout.  No wavelength-specific physics — this is the level of
abstraction required by a real-time CPS evaluation, which cares about
packet-level timing rather than waveform processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np

from skyshield.config import RadarConfig


@dataclass
class RadarPacket:
    node_id: int
    emit_time_ms: float          # when the radar generated the packet
    arrive_time_ms: float        # when the fusion plane receives it
    target_id: int
    position_m: Tuple[float, float, float]   # x, y, alt in metres
    velocity_mps: Tuple[float, float, float]
    snr_db: float
    valid: bool                  # False if the packet is a radar dropout
    meas_sigma_m: float          # measurement sigma used downstream


class RadarNode:
    """A single anchored PLFM-style radar node."""

    def __init__(self, node_id: int, position_km: Tuple[float, float],
                 cfg: RadarConfig):
        self.node_id = node_id
        self.position_km = position_km
        self.cfg = cfg
        self._last_emit_ms = -cfg.revisit_ms

    def in_coverage(self, target_km: Tuple[float, float]) -> bool:
        dx = target_km[0] - self.position_km[0]
        dy = target_km[1] - self.position_km[1]
        r = math.hypot(dx, dy)
        return r <= self.cfg.range_km_max

    def can_emit(self, now_ms: float) -> bool:
        return now_ms - self._last_emit_ms >= self.cfg.revisit_ms

    def _snr_db(self, range_km: float) -> float:
        # R^4 falloff; 25 dB at 1 km, floor at 5 dB at range_km_max.
        if range_km <= 0.01:
            return 40.0
        snr = 25.0 - 40.0 * math.log10(range_km)
        return max(snr, 4.0)

    def measurement_sigma(self, range_km: float) -> float:
        # Range-gate-driven measurement sigma in metres.  The sigma grows
        # roughly linearly with range (classical range-gate uncertainty)
        # plus a small dwell-time term.
        return max(2.0, 2.0 + 1.5 * range_km + 0.05 * self.cfg.dwell_ms)

    def observe(
        self,
        now_ms: float,
        target_id: int,
        target_pos_m: Tuple[float, float, float],
        target_vel_mps: Tuple[float, float, float],
        rng: np.random.Generator,
    ) -> Optional[RadarPacket]:
        target_km = (target_pos_m[0] / 1000.0, target_pos_m[1] / 1000.0)
        if not self.in_coverage(target_km):
            return None
        if not self.can_emit(now_ms):
            return None
        self._last_emit_ms = now_ms

        # Packet transit latency: truncated Gaussian at >= 0.5 ms.
        delay = max(0.5, rng.normal(self.cfg.packet_mean_ms,
                                    self.cfg.packet_jitter_ms))
        dropped = bool(rng.random() < self.cfg.dropout_rate)

        dx_km = target_km[0] - self.position_km[0]
        dy_km = target_km[1] - self.position_km[1]
        range_km = math.hypot(dx_km, dy_km)
        sigma_m = self.measurement_sigma(range_km)

        # Noisy measurement.
        noisy = (
            target_pos_m[0] + rng.normal(0.0, sigma_m),
            target_pos_m[1] + rng.normal(0.0, sigma_m),
            target_pos_m[2] + rng.normal(0.0, sigma_m),
        )

        return RadarPacket(
            node_id=self.node_id,
            emit_time_ms=now_ms,
            arrive_time_ms=now_ms + delay,
            target_id=target_id,
            position_m=noisy,
            velocity_mps=target_vel_mps,
            snr_db=self._snr_db(range_km),
            valid=not dropped,
            meas_sigma_m=sigma_m,
        )

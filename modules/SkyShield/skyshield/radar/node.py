"""PLFM-style radar node model.

Each ``RadarNode`` produces ``RadarPacket`` records on a fixed dwell
schedule.  Detection probability is a smooth function of slant range,
calibrated so that ``detection_pd_at_max`` is reached at ``range_km``.
A small Gaussian measurement-noise floor is applied so the downstream
Kalman filter has a meaningful innovation covariance.

The model is intentionally lightweight: SkyShield is a *systems*
artifact and the paper does not claim contributions to radar physics.
The numbers feeding into our timing budget come from the PLFM_RADAR
project referenced in `pressRequire/SkyShield/论文要求.md` §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry import Point


@dataclass
class RadarPacket:
    radar_id: int
    target_id: int
    t_emit_ms: float       # virtual time the dwell completed
    t_recv_ms: float       # virtual time the packet arrived at fusion
    measurement: np.ndarray  # (3,) noisy position estimate (m)
    snr_db: float
    pd: float              # instantaneous detection probability
    is_dropout: bool = False
    is_false_alarm: bool = False


@dataclass
class RadarNode:
    radar_id: int
    position: Point
    range_m: float
    azimuth_dwell_ms: float
    pd_at_max: float
    false_alarm_per_min: float
    measurement_noise_r: float
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def detection_probability(self, target: Point) -> float:
        d = self.position.slant_distance(target)
        if d > self.range_m:
            return 0.0
        # smooth roll-off: pd = 1 at d=0 -> pd_at_max at d=range
        x = d / self.range_m
        return float(1.0 - (1.0 - self.pd_at_max) * (x ** 2))

    def snr_db(self, target: Point) -> float:
        """Toy SNR: 30 dB at range 0, falls off as 40 log10(range/range_max)."""
        d = max(1.0, self.position.slant_distance(target))
        rel = max(1e-3, d / self.range_m)
        snr = 30.0 - 40.0 * np.log10(rel)
        return float(snr)

    def measure(
        self,
        target_id: int,
        target: Point,
        t_emit_ms: float,
        link_jitter_ms_std: float,
        link_dropout_pct: float,
    ) -> Optional[RadarPacket]:
        pd = self.detection_probability(target)
        if pd <= 0.0:
            return None
        # Bernoulli detection draw
        if self.rng.random() > pd:
            return None
        # measurement: ground truth + Gaussian noise scaled by 1/SNR
        snr = self.snr_db(target)
        noise_sigma = self.measurement_noise_r * (1.0 + max(0.0, (20.0 - snr) / 20.0))
        meas = target.as_array() + self.rng.normal(0.0, noise_sigma, 3)
        # link transport delay: dwell-driven base + jitter
        jitter = abs(self.rng.normal(0.0, link_jitter_ms_std))
        t_recv_ms = t_emit_ms + jitter
        is_dropout = self.rng.random() * 100.0 < link_dropout_pct
        if is_dropout:
            return RadarPacket(
                radar_id=self.radar_id,
                target_id=target_id,
                t_emit_ms=t_emit_ms,
                t_recv_ms=t_recv_ms,
                measurement=meas,
                snr_db=snr,
                pd=pd,
                is_dropout=True,
            )
        return RadarPacket(
            radar_id=self.radar_id,
            target_id=target_id,
            t_emit_ms=t_emit_ms,
            t_recv_ms=t_recv_ms,
            measurement=meas,
            snr_db=snr,
            pd=pd,
        )

    def maybe_emit_false_alarm(self, t_ms: float, dt_ms: float) -> Optional[RadarPacket]:
        rate = self.false_alarm_per_min / 60_000.0  # per ms
        p = rate * dt_ms
        if self.rng.random() < p:
            jitter = abs(self.rng.normal(0.0, 1.0))
            return RadarPacket(
                radar_id=self.radar_id,
                target_id=-1,
                t_emit_ms=t_ms,
                t_recv_ms=t_ms + jitter,
                measurement=self.position.as_array() + self.rng.normal(0, 30.0, 3),
                snr_db=10.0,
                pd=0.0,
                is_false_alarm=True,
            )
        return None

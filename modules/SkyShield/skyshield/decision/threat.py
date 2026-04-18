"""Threat scoring for a confirmed track.

The score is a bounded combination of (i) closing speed toward a
protected zone, (ii) altitude band fit, (iii) distance to the nearest
no-fly polygon.  This intentionally skips classifier-level semantics:
RTSS cares about *timing* rather than perception IP.  The score is in
[0, 1] and clipped.
"""
from __future__ import annotations

from typing import Sequence, Tuple
import math

import numpy as np

from skyshield.config import CityConfig


def _closing_speed(pos_m: Sequence[float], vel_mps: Sequence[float],
                   target_km: Tuple[float, float]) -> float:
    tgt_m = (target_km[0] * 1000.0, target_km[1] * 1000.0)
    dx = tgt_m[0] - pos_m[0]
    dy = tgt_m[1] - pos_m[1]
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return math.hypot(vel_mps[0], vel_mps[1])
    # Project velocity onto the direction-to-target.
    u = (dx / dist, dy / dist)
    return vel_mps[0] * u[0] + vel_mps[1] * u[1]


def score_threat(
    pos_m: Sequence[float],
    vel_mps: Sequence[float],
    city: CityConfig,
    target_class_conf: float = 1.0,
) -> float:
    # Altitude band: 30-300 m is "suspicious" for counter-UAV work.
    alt = pos_m[2]
    band = 1.0 if 30.0 <= alt <= 300.0 else max(0.0, 1.0 - abs(alt - 160.0) / 400.0)

    # Distance to nearest no-fly zone.
    if city.no_fly_zones:
        nearest = min(
            max(0.0, math.hypot(pos_m[0] / 1000.0 - z.center_km[0],
                                pos_m[1] / 1000.0 - z.center_km[1]) - z.radius_km)
            for z in city.no_fly_zones
        )
    else:
        nearest = 5.0
    dist_term = max(0.0, 1.0 - nearest / 2.5)

    # Closing speed toward the nearest zone (m/s); turn into [0, 1].
    if city.no_fly_zones:
        tgt = min(city.no_fly_zones,
                  key=lambda z: math.hypot(pos_m[0] / 1000.0 - z.center_km[0],
                                           pos_m[1] / 1000.0 - z.center_km[1]))
        closing = max(0.0, _closing_speed(pos_m, vel_mps, tgt.center_km))
    else:
        closing = math.hypot(vel_mps[0], vel_mps[1])
    closing_term = 1.0 / (1.0 + math.exp(-(closing - 20.0) / 6.0))

    # A confirmed track *inside the defended bbox* with valid class is
    # inherently suspicious.  Weights are tuned so that (i) a confirmed
    # low-altitude intruder over the district scores > 0.55 and (ii)
    # a far / high / slow contact with weak classification stays < 0.55.
    score = (0.35 * band + 0.15 * dist_term + 0.20 * closing_term
             + 0.30 * target_class_conf)
    return float(np.clip(score, 0.0, 1.0))

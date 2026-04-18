"""Closed-form interceptor kinematics model.

We do not simulate 6-DOF flight.  The paper's timing claims only need
a time-to-intercept estimate and a hit-probability draw, both of
which are functions of range, relative velocity and target maneuver
state.  The nominal parameters come straight from the Counter-UAV
specification sheet in the requirements (350 km/h peak, 200 km/h
cruise, 22 m/s^2 acceleration, 3.5 min endurance at full payload).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np

from skyshield.config import InterceptorConfig


@dataclass
class EngagementResult:
    launched: bool
    hit: bool
    shot_down: bool
    reaction_ms: float
    time_to_intercept_ms: float
    terminal_closing_mps: float
    reason: str = ""


class InterceptorModel:
    def __init__(self, cfg: InterceptorConfig):
        self.cfg = cfg

    def time_to_intercept_ms(
        self,
        launch_point_km: Sequence[float],
        target_pos_m: Sequence[float],
        target_vel_mps: Sequence[float],
    ) -> float:
        """Approximate the time to first valid intercept geometry.

        The interceptor accelerates at ``acc_mps2`` up to ``max_speed_mps``
        along the line-of-sight; the target keeps its current velocity.
        We solve the simple 1-D closing-range equation.
        """
        start_m = (launch_point_km[0] * 1000.0, launch_point_km[1] * 1000.0)
        dx = target_pos_m[0] - start_m[0]
        dy = target_pos_m[1] - start_m[1]
        r = math.hypot(dx, dy)
        if r < 1.0:
            return 50.0
        target_speed_along = abs(
            (target_vel_mps[0] * dx + target_vel_mps[1] * dy) / r
        )
        closing = max(self.cfg.max_speed_mps - target_speed_along, 20.0)

        # Acceleration phase duration (time to reach max speed).
        t_acc = self.cfg.max_speed_mps / self.cfg.acc_mps2
        dist_acc = 0.5 * self.cfg.acc_mps2 * t_acc ** 2
        if dist_acc >= r:
            # Still in acceleration phase at intercept.
            t = math.sqrt(2.0 * r / self.cfg.acc_mps2)
            return 1000.0 * t
        remaining = r - dist_acc
        t_cruise = remaining / closing
        return 1000.0 * (t_acc + t_cruise)

    def engage(
        self,
        launch_point_km: Sequence[float],
        target_pos_m: Sequence[float],
        target_vel_mps: Sequence[float],
        target_maneuvering: bool,
        rng: np.random.Generator,
    ) -> EngagementResult:
        reaction_ms = float(rng.normal(90.0, 22.0))
        reaction_ms = max(reaction_ms, 40.0)

        tti = self.time_to_intercept_ms(
            launch_point_km, target_pos_m, target_vel_mps
        )
        tti_jitter = float(rng.normal(0.0, 0.04 * tti))
        tti_total = max(80.0, tti + tti_jitter)

        if tti_total > self.cfg.endurance_s * 1000.0:
            return EngagementResult(
                launched=True, hit=False, shot_down=False,
                reaction_ms=reaction_ms, time_to_intercept_ms=tti_total,
                terminal_closing_mps=0.0, reason="endurance_exceeded",
            )

        p_hit = self.cfg.hit_prob_nominal
        if target_maneuvering:
            p_hit = self.cfg.hit_prob_under_maneuver

        hit = bool(rng.random() < p_hit)
        if hit:
            # Conditional on hit, 70% outright shoot-down (matches sorties 4/5/9/10).
            shot_down = bool(rng.random() < 0.70)
        else:
            shot_down = False

        # Terminal closing speed recovers the line-of-sight closing rate,
        # but with the interceptor's full velocity vector at impact.
        dx = target_pos_m[0] - launch_point_km[0] * 1000.0
        dy = target_pos_m[1] - launch_point_km[1] * 1000.0
        r = max(1.0, math.hypot(dx, dy))
        closing = (self.cfg.max_speed_mps
                   - abs((target_vel_mps[0] * dx + target_vel_mps[1] * dy) / r))
        closing = max(closing, self.cfg.cruise_speed_mps)

        return EngagementResult(
            launched=True, hit=hit, shot_down=shot_down,
            reaction_ms=reaction_ms, time_to_intercept_ms=tti_total,
            terminal_closing_mps=float(closing), reason="engaged",
        )

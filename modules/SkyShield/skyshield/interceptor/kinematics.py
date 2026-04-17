"""Interceptor kinematics + hit-probability model.

The 1.8 kg airframe / 350 km/h sustain / 3.5 min endurance numbers
come straight from the field-test spec sheet in
``pressRequire/SkyShield/论文要求.md`` §3.1.  Hit probability degrades
gracefully as a function of (a) closing speed, (b) target manoeuvre
load, and (c) tracking covariance at the terminal frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class InterceptOutcome(str, Enum):
    HIT_SHOT_DOWN = "hit_shot_down"
    HIT_NOT_SHOT_DOWN = "hit_not_shot_down"
    TARGET_LOST = "target_lost"
    ABORTED = "aborted"
    MISS = "miss"


@dataclass
class InterceptResult:
    outcome: InterceptOutcome
    hit_time_s: float
    hit_height_m: float
    terminal_strike_kmh: float
    closing_speed_kmh: float
    miss_distance_m: float


@dataclass
class InterceptorKinematics:
    max_speed_kmh: float = 350.0
    cruise_speed_kmh: float = 200.0
    endurance_s: float = 210.0
    hit_prob_base: float = 0.80
    rng: np.random.Generator = None  # type: ignore

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(0)

    def predict(
        self,
        target_speed_kmh: float,
        target_height_m: float,
        target_maneuver_g: float,
        track_cov_trace: float,
        intercept_distance_m: float,
        already_aborted: bool = False,
    ) -> InterceptResult:
        if already_aborted:
            return InterceptResult(
                outcome=InterceptOutcome.ABORTED,
                hit_time_s=0.0,
                hit_height_m=target_height_m,
                terminal_strike_kmh=0.0,
                closing_speed_kmh=0.0,
                miss_distance_m=0.0,
            )

        # closing speed model
        closing_kmh = float(self.max_speed_kmh + 0.4 * target_speed_kmh)
        # time-to-go in seconds (range / closing speed)
        ttg = max(2.0, intercept_distance_m / max(1.0, closing_kmh / 3.6))

        # hit probability degraded by manoeuvre and covariance
        # nominal base from spec sheet (0.80 = 80% strike accuracy)
        hp = self.hit_prob_base
        hp -= 0.08 * max(0.0, target_maneuver_g - 0.5)
        hp -= 0.0009 * max(0.0, track_cov_trace - 50.0)
        hp = float(np.clip(hp, 0.05, 0.97))
        roll = self.rng.random()
        if roll > hp:
            # missed entirely; either lost or genuine miss depending on cov
            if track_cov_trace > 200.0:
                return InterceptResult(
                    outcome=InterceptOutcome.TARGET_LOST,
                    hit_time_s=ttg,
                    hit_height_m=target_height_m,
                    terminal_strike_kmh=0.0,
                    closing_speed_kmh=closing_kmh,
                    miss_distance_m=float(self.rng.uniform(8.0, 35.0)),
                )
            return InterceptResult(
                outcome=InterceptOutcome.MISS,
                hit_time_s=ttg,
                hit_height_m=target_height_m,
                terminal_strike_kmh=closing_kmh * 0.7,
                closing_speed_kmh=closing_kmh,
                miss_distance_m=float(self.rng.uniform(2.5, 9.0)),
            )

        # hit -> decide shoot-down vs glancing impact
        # P(shot_down | hit) is 0.65 nominal, scaled by closing speed
        p_shoot = 0.65 + 0.10 * (closing_kmh - 300.0) / 200.0
        p_shoot = float(np.clip(p_shoot, 0.30, 0.95))
        outcome = (
            InterceptOutcome.HIT_SHOT_DOWN
            if self.rng.random() < p_shoot
            else InterceptOutcome.HIT_NOT_SHOT_DOWN
        )
        return InterceptResult(
            outcome=outcome,
            hit_time_s=ttg,
            hit_height_m=float(target_height_m + self.rng.normal(0.0, 1.5)),
            terminal_strike_kmh=float(closing_kmh + self.rng.normal(0.0, 5.0)),
            closing_speed_kmh=closing_kmh,
            miss_distance_m=float(abs(self.rng.normal(0.0, 0.8))),
        )

from __future__ import annotations

import numpy as np

from skyshield.config import InterceptorConfig
from skyshield.interceptor.kinematics import InterceptorModel


def _cfg():
    return InterceptorConfig(
        mass_kg=1.8, max_speed_mps=97.2, cruise_speed_mps=55.6,
        endurance_s=210.0, acc_mps2=22.0,
        hit_prob_nominal=0.8, hit_prob_under_maneuver=0.52,
        base_km=(10.0, 7.5),
    )


def test_tti_within_expected_range():
    m = InterceptorModel(_cfg())
    tti = m.time_to_intercept_ms(
        launch_point_km=(10.0, 7.5),
        target_pos_m=(10500.0, 7500.0, 120.0),    # 500 m off-axis
        target_vel_mps=(30.0, 0.0, 0.0),
    )
    assert 3500.0 < tti < 18000.0


def test_maneuvering_target_reduces_hit_prob():
    m = InterceptorModel(_cfg())
    rng = np.random.default_rng(0)
    hits_clean = sum(
        m.engage((10.0, 7.5), (10800.0, 7500.0, 120.0),
                 (30.0, 0.0, 0.0), False, rng).hit
        for _ in range(400)
    )
    rng = np.random.default_rng(0)
    hits_maneuver = sum(
        m.engage((10.0, 7.5), (10800.0, 7500.0, 120.0),
                 (30.0, 0.0, 0.0), True, rng).hit
        for _ in range(400)
    )
    assert hits_maneuver < hits_clean


def test_endurance_exceeded_marked():
    cfg = _cfg()
    cfg.endurance_s = 0.1
    m = InterceptorModel(cfg)
    rng = np.random.default_rng(0)
    er = m.engage((0.0, 0.0), (8000.0, 0.0, 120.0),
                  (30.0, 0.0, 0.0), False, rng)
    assert er.reason == "endurance_exceeded"
    assert not er.hit

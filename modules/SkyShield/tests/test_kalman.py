"""Unit tests for the constant-velocity Kalman tracker."""
from __future__ import annotations

import numpy as np

from skyshield.config import TrackerConfig
from skyshield.tracker.kalman import KalmanTracker


def _cfg() -> TrackerConfig:
    return TrackerConfig(
        process_noise=0.8, meas_noise_m=10.0,
        confirm_m_of_n=(3, 5), gate_sigma=3.0, degraded_mode=True,
    )


def test_kalman_converges_on_constant_velocity():
    cfg = _cfg()
    kt = KalmanTracker(cfg)
    x = 0.0
    vx = 30.0
    t = 0.0
    # Observe 20 measurements at 35 ms revisit with 1-sigma = 10 m noise.
    rng = np.random.default_rng(0)
    for _ in range(20):
        t += 35.0
        x += vx * 0.035
        pos = np.array([x + rng.normal(0.0, 5.0), 0.0, 120.0])
        vel = np.array([vx, 0.0, 0.0])
        kt.update(track_id=7, t_ms=t, pos=pos, vel=vel, meas_sigma_m=5.0)
    st = kt.get(7)
    assert st is not None and st.valid
    assert abs(st.x[3] - vx) < 4.0        # velocity estimate tight
    assert abs(st.x[0] - x) < 8.0         # position estimate within ~1 sigma


def test_kalman_coasts_then_invalidates():
    cfg = _cfg()
    kt = KalmanTracker(cfg)
    kt.update(5, 0.0, np.array([0.0, 0.0, 100.0]),
              np.array([30.0, 0.0, 0.0]), meas_sigma_m=5.0)
    st = kt.update(5, 300.0, pos=None, vel=None, meas_sigma_m=5.0)
    assert st.valid                            # still coasting
    st = kt.update(5, 800.0, pos=None, vel=None, meas_sigma_m=5.0)
    assert not st.valid

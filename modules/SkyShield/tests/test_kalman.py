"""Tests for the Kalman / IMM trackers."""

from __future__ import annotations

import numpy as np
import pytest

from skyshield.tracker import IMMTracker, KalmanCV


def _line(steps: int, dt: float, vx: float, noise: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    truth = np.zeros((steps, 3))
    obs = np.zeros((steps, 3))
    for k in range(steps):
        truth[k, 0] = vx * k * dt
        obs[k] = truth[k] + rng.normal(0.0, noise, size=3)
    return truth, obs


def test_kalman_cv_converges_to_constant_velocity():
    truth, obs = _line(steps=80, dt=0.05, vx=30.0, noise=2.5)
    f = KalmanCV(q=1.0, r=6.0)
    last_pos = None
    for k, z in enumerate(obs):
        f.predict(0.05)
        last_pos, _ = f.update(z)
    assert f.initialized
    err_pos = np.linalg.norm(last_pos[:3] - truth[-1])
    err_vel = abs(f.velocity()[0] - 30.0)
    assert err_pos < 4.0, f"final pos error {err_pos:.2f} m too large"
    assert err_vel < 4.0, f"final vx error {err_vel:.2f} m/s too large"


def test_kalman_cv_covariance_decreases_after_updates():
    truth, obs = _line(steps=40, dt=0.05, vx=20.0, noise=2.0, seed=11)
    f = KalmanCV(q=1.0, r=4.0)
    f.update(obs[0])
    initial_trace = float(np.trace(f.position_cov()))
    for z in obs[1:]:
        f.predict(0.05)
        f.update(z)
    final_trace = float(np.trace(f.position_cov()))
    assert final_trace < initial_trace, "covariance should shrink with measurements"
    assert final_trace < 12.0, f"covariance trace {final_trace:.2f} unreasonably large"


def test_imm_tracks_constant_velocity_target():
    truth, obs = _line(steps=80, dt=0.05, vx=25.0, noise=3.0, seed=7)
    imm = IMMTracker(q_cv=1.0, q_man=8.0, r=8.0)
    last_pos = None
    for z in obs:
        last_pos, _ = imm.step(z, dt=0.05)
    err = float(np.linalg.norm(last_pos - truth[-1]))
    assert err < 8.0, f"IMM track error {err:.2f} m too large for CV target"
    # Both model weights should be valid probabilities summing to 1.
    assert abs(imm.weights.sum() - 1.0) < 1e-6
    assert (imm.weights >= 0).all()


def test_imm_velocity_estimate_aligned_with_truth():
    truth, obs = _line(steps=80, dt=0.05, vx=22.0, noise=2.0, seed=13)
    imm = IMMTracker(q_cv=1.0, q_man=8.0, r=4.0)
    for z in obs:
        imm.step(z, dt=0.05)
    vel = imm.velocity()
    assert abs(vel[0] - 22.0) < 5.0, f"vx estimate off: {vel[0]:.2f}"

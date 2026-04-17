"""Constant-velocity and IMM-style Kalman filters in 3D position.

The state is ``[x, y, z, vx, vy, vz]``.  Process noise ``Q`` and
measurement noise ``R`` are scalar inputs that are broadcast to the
right shape.  We deliberately keep the implementation small (~80 lines)
because SkyShield does not claim contributions to estimation theory:
the filter is here so that the perception link produces a consistent
``FusedTrack`` covariance for the deadline-aware decision module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _F(dt: float) -> np.ndarray:
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def _Q(dt: float, q: float) -> np.ndarray:
    # Standard CV process noise covariance scaled by q.
    Q = np.zeros((6, 6))
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    for i in range(3):
        Q[i, i] = dt4 / 4.0 * q
        Q[i, 3 + i] = dt3 / 2.0 * q
        Q[3 + i, i] = dt3 / 2.0 * q
        Q[3 + i, 3 + i] = dt2 * q
    return Q


@dataclass
class KalmanCV:
    q: float = 1.5
    r: float = 6.0
    state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    cov: np.ndarray = field(default_factory=lambda: np.eye(6) * 100.0)
    initialized: bool = False

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            return self.state.copy(), self.cov.copy()
        F = _F(dt)
        Q = _Q(dt, self.q)
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        return self.state.copy(), self.cov.copy()

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            self.state[:3] = z
            self.state[3:] = 0.0
            self.cov = np.eye(6) * (self.r * 4)
            self.initialized = True
            return self.state.copy(), self.cov.copy()
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        R = np.eye(3) * self.r
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        y = z - H @ self.state
        self.state = self.state + K @ y
        self.cov = (np.eye(6) - K @ H) @ self.cov
        return self.state.copy(), self.cov.copy()

    def position(self) -> np.ndarray:
        return self.state[:3].copy()

    def velocity(self) -> np.ndarray:
        return self.state[3:].copy()

    def position_cov(self) -> np.ndarray:
        return self.cov[:3, :3].copy()


@dataclass
class IMMTracker:
    """A two-model IMM that mixes a CV filter and a higher-q maneuver filter."""
    q_cv: float = 1.5
    q_man: float = 8.0
    r: float = 6.0
    p_trans: tuple[float, float] = (0.95, 0.85)
    cv: KalmanCV = field(default=None)  # type: ignore
    man: KalmanCV = field(default=None)  # type: ignore
    weights: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))

    def __post_init__(self) -> None:
        if self.cv is None:
            self.cv = KalmanCV(q=self.q_cv, r=self.r)
        if self.man is None:
            self.man = KalmanCV(q=self.q_man, r=self.r)

    def step(self, z: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        for f in (self.cv, self.man):
            f.predict(dt)
        like = []
        for f in (self.cv, self.man):
            f.update(z)
            innov_cov = np.eye(3) * f.r + f.position_cov()
            innov = z - f.position()
            try:
                detS = np.linalg.det(innov_cov)
                like.append(np.exp(-0.5 * innov.T @ np.linalg.inv(innov_cov) @ innov)
                            / max(1e-9, np.sqrt((2 * np.pi) ** 3 * detS)))
            except Exception:  # pragma: no cover - guarded by detS test
                like.append(1e-9)
        like = np.array(like) + 1e-9
        # transition
        keep = np.array([self.p_trans[0], self.p_trans[1]])
        priors = self.weights * keep + self.weights[::-1] * (1.0 - keep)
        post = priors * like
        post /= post.sum()
        self.weights = post
        # mixed estimate
        pos = self.weights[0] * self.cv.position() + self.weights[1] * self.man.position()
        cov = (
            self.weights[0] * self.cv.position_cov()
            + self.weights[1] * self.man.position_cov()
        )
        return pos, cov

    def velocity(self) -> np.ndarray:
        return self.weights[0] * self.cv.velocity() + self.weights[1] * self.man.velocity()

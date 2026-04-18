"""Constant-velocity Kalman filter used downstream of fusion to smooth
fused tracks and to carry the *confirmed* motion state into the
decision plane.  In degraded mode (packet dropout bursts) the filter
coasts on dynamics for a bounded window before invalidating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from skyshield.config import TrackerConfig


@dataclass
class KalmanState:
    t_ms: float
    x: np.ndarray           # [px, py, pz, vx, vy, vz]
    P: np.ndarray           # 6x6 covariance
    valid: bool = True
    coast_ms: float = 0.0   # time since last *valid* measurement


class KalmanTracker:
    """One CV Kalman filter per fused track."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._states: dict[int, KalmanState] = {}

    def _init_state(self, track_id: int, t_ms: float,
                    pos: np.ndarray, vel: np.ndarray) -> KalmanState:
        x = np.concatenate([pos, vel])
        P = np.eye(6) * (self.cfg.meas_noise_m ** 2)
        st = KalmanState(t_ms=t_ms, x=x, P=P)
        self._states[track_id] = st
        return st

    def _predict(self, st: KalmanState, t_ms: float) -> None:
        dt = max(1e-3, (t_ms - st.t_ms) / 1000.0)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        q = self.cfg.process_noise ** 2
        Q = np.zeros((6, 6))
        # Continuous white-noise acceleration model (discrete form).
        Q[0, 0] = Q[1, 1] = Q[2, 2] = (dt ** 3) / 3.0 * q
        Q[3, 3] = Q[4, 4] = Q[5, 5] = dt * q
        Q[0, 3] = Q[3, 0] = (dt ** 2) / 2.0 * q
        Q[1, 4] = Q[4, 1] = (dt ** 2) / 2.0 * q
        Q[2, 5] = Q[5, 2] = (dt ** 2) / 2.0 * q

        st.x = F @ st.x
        st.P = F @ st.P @ F.T + Q
        st.t_ms = t_ms

    def update(
        self,
        track_id: int,
        t_ms: float,
        pos: Optional[np.ndarray],
        vel: Optional[np.ndarray],
        meas_sigma_m: float,
    ) -> KalmanState:
        st = self._states.get(track_id)
        if st is None:
            if pos is None or vel is None:
                raise ValueError("Cannot bootstrap tracker with missing data.")
            return self._init_state(track_id, t_ms, np.asarray(pos), np.asarray(vel))

        if pos is None:
            # Coast; book the elapsed time *before* predicting so the
            # running coast counter is monotone.
            coast_delta = max(0.0, t_ms - st.t_ms)
            self._predict(st, t_ms)
            st.coast_ms += coast_delta
            if not self.cfg.degraded_mode or st.coast_ms > 400.0:
                st.valid = False
            return st

        self._predict(st, t_ms)

        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * max(meas_sigma_m, 2.0) ** 2
        z = np.asarray(pos)
        y = z - H @ st.x
        S = H @ st.P @ H.T + R
        K = st.P @ H.T @ np.linalg.inv(S)
        st.x = st.x + K @ y
        st.P = (np.eye(6) - K @ H) @ st.P
        st.coast_ms = 0.0
        st.valid = True
        return st

    def get(self, track_id: int) -> Optional[KalmanState]:
        return self._states.get(track_id)

    def drop(self, track_id: int) -> None:
        self._states.pop(track_id, None)

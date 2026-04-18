"""Track-level multi-radar fusion with covariance-weighted updates and
explicit handoff bookkeeping.  The fusion plane is intentionally kept
coarse because the paper's real-time contribution is the *timing* of
the fused track, not its estimation accuracy per se."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from skyshield.radar.node import RadarPacket
from skyshield.config import RadarConfig


@dataclass
class FusedTrack:
    target_id: int
    last_update_ms: float
    first_update_ms: float
    position_m: Tuple[float, float, float]
    velocity_mps: Tuple[float, float, float]
    covariance: np.ndarray     # 3x3 position covariance
    contributing_nodes: List[int] = field(default_factory=list)
    handoff_latency_ms: float = 0.0


class MultiRadarFuser:
    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        self._tracks: Dict[int, FusedTrack] = {}
        self._last_node: Dict[int, int] = {}

    def ingest(self, packet: RadarPacket) -> Optional[FusedTrack]:
        if not packet.valid:
            return None
        tid = packet.target_id

        pos = np.asarray(packet.position_m, dtype=float)
        # Diagonal covariance proportional to meas_sigma squared.
        sigma = packet.meas_sigma_m
        meas_cov = np.eye(3) * sigma * sigma

        track = self._tracks.get(tid)
        if track is None or not self.cfg.fusion_enabled:
            self._tracks[tid] = FusedTrack(
                target_id=tid,
                last_update_ms=packet.arrive_time_ms,
                first_update_ms=packet.arrive_time_ms,
                position_m=tuple(pos),
                velocity_mps=tuple(packet.velocity_mps),
                covariance=meas_cov,
                contributing_nodes=[packet.node_id],
            )
            self._last_node[tid] = packet.node_id
            return self._tracks[tid]

        # Covariance-weighted update (essentially Kalman without dynamics).
        P = track.covariance
        K = P @ np.linalg.inv(P + meas_cov)
        fused_pos = np.asarray(track.position_m) + K @ (pos - np.asarray(track.position_m))
        fused_cov = (np.eye(3) - K) @ P

        # Velocity: running average weighted by time gap (simple + stable).
        v_prev = np.asarray(track.velocity_mps)
        v_new = np.asarray(packet.velocity_mps)
        dt_ms = max(1e-3, packet.arrive_time_ms - track.last_update_ms)
        blend = min(0.5, 20.0 / dt_ms)  # recent data weighted higher for fast jitter
        fused_v = (1.0 - blend) * v_prev + blend * v_new

        handoff = 0.0
        if self._last_node.get(tid) != packet.node_id:
            handoff = dt_ms  # time since the last update from a *different* node
            self._last_node[tid] = packet.node_id

        track.position_m = tuple(fused_pos)
        track.velocity_mps = tuple(fused_v)
        track.covariance = fused_cov
        track.last_update_ms = packet.arrive_time_ms
        track.handoff_latency_ms = handoff
        if packet.node_id not in track.contributing_nodes:
            track.contributing_nodes.append(packet.node_id)

        return track

    def all_tracks(self) -> Dict[int, FusedTrack]:
        return dict(self._tracks)

    def drop(self, target_id: int) -> None:
        self._tracks.pop(target_id, None)
        self._last_node.pop(target_id, None)

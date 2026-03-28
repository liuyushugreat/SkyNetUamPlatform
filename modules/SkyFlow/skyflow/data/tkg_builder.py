"""Temporal Knowledge Graph construction — Algorithm 1 in the paper.

Constructs typed entity-relation-time graphs G_t = (V_t, E_t, R, τ)
from four parallel data streams (Section 3.1):
  - ADS-B telemetry → UAV nodes with 23-dim state vector (Eq. 1)
  - Flight plans → corridor reservation edges (shares_corridor)
  - Weather grid → downwind influence edges (is_downwind_of)
  - Corridor log → restricted zone proximity edges (is_restricted_by)

Six relation types R (Section 3.2): approaches, conflicts_with,
shares_corridor, is_downwind_of, has_reserved, is_restricted_by.

Each edge carries elapsed time δ since last observation, fed to
the sinusoidal temporal encoding φ(δ) in Eq. (2).

Designed to execute in < 12 ms on benchmark hardware (Table 7).

Reference: Section 3 and Algorithm 1 in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


ENTITY_TYPES = {"uav": 0, "sector": 1, "weather": 2, "restricted": 3}

RELATION_VOCAB = {
    "approaches": 0,
    "conflicts_with": 1,
    "shares_corridor": 2,
    "is_downwind_of": 3,
    "has_reserved": 4,
    "is_restricted_by": 5,
}


@dataclass
class AirspaceState:
    """Raw airspace state at a single epoch."""

    uav_positions: np.ndarray       # (N_uav, 3) xyz in meters
    uav_velocities: np.ndarray      # (N_uav, 3)
    uav_headings: np.ndarray        # (N_uav,) radians
    uav_battery: np.ndarray         # (N_uav,) fraction [0,1]
    uav_priority: np.ndarray        # (N_uav,) int class
    uav_avoiding: np.ndarray        # (N_uav,) bool
    sector_occupancy: np.ndarray    # (N_sec, 8)
    weather_cells: np.ndarray       # (N_wx, 12)
    restricted_zones: np.ndarray    # (N_rz, 8)
    corridor_reservations: List[Tuple[int, int, float, float]]
    epoch_time: float
    uav_heading_rates: Optional[np.ndarray] = None    # (N_uav,) rad/s
    uav_accelerations: Optional[np.ndarray] = None    # (N_uav, 3) m/s²
    uav_battery_rates: Optional[np.ndarray] = None    # (N_uav,) discharge rate
    uav_corridor_ids: Optional[np.ndarray] = None     # (N_uav,) corridor assignment
    uav_local_wind: Optional[np.ndarray] = None       # (N_uav, 3) local wind estimate
    uav_gps_dop: Optional[np.ndarray] = None          # (N_uav,) GPS dilution of precision


@dataclass
class TKGSnapshot:
    """A single temporal knowledge graph snapshot ready for TR-GAT."""

    node_features: torch.Tensor
    node_types: torch.Tensor
    edge_indices: Dict[int, torch.Tensor]
    edge_deltas: Dict[int, torch.Tensor]
    num_uavs: int
    num_nodes: int

    conflict_pairs: Optional[torch.Tensor] = None
    conflict_labels: Optional[torch.Tensor] = None


class TKGBuilder:
    """Builds TKG snapshots from raw airspace state."""

    def __init__(
        self,
        approach_cpa_h: float = 80.0,
        approach_cpa_v: float = 15.0,
        approach_lookahead: float = 60.0,
        corridor_lookahead: float = 120.0,
        weather_radius: float = 400.0,
        feature_dim: int = 23,
    ):
        self.approach_cpa_h = approach_cpa_h
        self.approach_cpa_v = approach_cpa_v
        self.approach_lookahead = approach_lookahead
        self.corridor_lookahead = corridor_lookahead
        self.weather_radius = weather_radius
        self.feature_dim = feature_dim

        self._last_edge_times: Dict[int, Dict[Tuple[int, int], float]] = {
            r: {} for r in range(6)
        }

    def build(
        self, state: AirspaceState, device: torch.device = torch.device("cpu")
    ) -> TKGSnapshot:
        """Construct a TKG snapshot from raw airspace state."""
        n_uav = state.uav_positions.shape[0]
        n_sec = state.sector_occupancy.shape[0]
        n_wx = state.weather_cells.shape[0]
        n_rz = state.restricted_zones.shape[0]
        n_total = n_uav + n_sec + n_wx + n_rz

        node_features = self._build_node_features(state, n_total, n_uav, n_sec, n_wx, n_rz)
        node_types = self._build_node_types(n_uav, n_sec, n_wx, n_rz)

        edge_indices, edge_deltas = self._build_edges(state, n_uav, n_sec, n_wx, n_rz)

        ei_tensors = {
            r: torch.tensor(edges, dtype=torch.long, device=device)
            for r, edges in edge_indices.items()
            if len(edges[0]) > 0
        }
        ed_tensors = {
            r: torch.tensor(deltas, dtype=torch.float32, device=device)
            for r, deltas in edge_deltas.items()
            if len(deltas) > 0
        }

        return TKGSnapshot(
            node_features=torch.tensor(node_features, dtype=torch.float32, device=device),
            node_types=torch.tensor(node_types, dtype=torch.long, device=device),
            edge_indices=ei_tensors,
            edge_deltas=ed_tensors,
            num_uavs=n_uav,
            num_nodes=n_total,
        )

    def _build_node_features(
        self,
        state: AirspaceState,
        n_total: int,
        n_uav: int,
        n_sec: int,
        n_wx: int,
        n_rz: int,
    ) -> np.ndarray:
        """Build the 23-dim UAV feature vector per Eq. (1) in the paper:
        [x,y,z, vx,vy,vz, ψ,ψ̇, ax,ay,az, b,ḃ, p, c_id,
         wx,wy,wz, σ_gps, n_nbr, d_min, t_cpa, f_avoid]
        """
        feat = np.zeros((n_total, self.feature_dim), dtype=np.float32)

        heading_rates = state.uav_heading_rates if state.uav_heading_rates is not None else np.zeros(n_uav, dtype=np.float32)
        accelerations = state.uav_accelerations if state.uav_accelerations is not None else np.zeros((n_uav, 3), dtype=np.float32)
        battery_rates = state.uav_battery_rates if state.uav_battery_rates is not None else np.full(n_uav, -0.0001, dtype=np.float32)
        corridor_ids = state.uav_corridor_ids if state.uav_corridor_ids is not None else np.zeros(n_uav, dtype=np.float32)
        local_wind = state.uav_local_wind if state.uav_local_wind is not None else np.zeros((n_uav, 3), dtype=np.float32)
        gps_dop = state.uav_gps_dop if state.uav_gps_dop is not None else np.full(n_uav, 2.5, dtype=np.float32)

        for i in range(n_uav):
            feat[i, 0:3] = state.uav_positions[i]                # x, y, z
            feat[i, 3:6] = state.uav_velocities[i]               # vx, vy, vz
            feat[i, 6] = state.uav_headings[i]                   # ψ
            feat[i, 7] = heading_rates[i]                         # ψ̇
            feat[i, 8:11] = accelerations[i]                      # ax, ay, az
            feat[i, 11] = state.uav_battery[i]                   # b
            feat[i, 12] = battery_rates[i]                        # ḃ
            feat[i, 13] = state.uav_priority[i]                  # p
            feat[i, 14] = corridor_ids[i]                         # c_id
            feat[i, 15:18] = local_wind[i]                        # wx, wy, wz
            feat[i, 18] = gps_dop[i]                              # σ_gps

        if n_uav > 0:
            positions = state.uav_positions[:n_uav]
            for i in range(n_uav):
                diffs = positions - positions[i]
                dists = np.sqrt((diffs[:, 0] ** 2) + (diffs[:, 1] ** 2) + 1e-12)
                dists[i] = np.inf
                nbr_count = np.sum(dists < self.approach_cpa_h)
                feat[i, 19] = nbr_count                          # n_nbr
                nearest = np.argmin(dists)
                feat[i, 20] = dists[nearest]                      # d_min
                dp = diffs[nearest]
                dv = state.uav_velocities[nearest] - state.uav_velocities[i]
                dvdv = np.dot(dv, dv)
                if dvdv > 1e-8:
                    t_cpa = max(0.0, -np.dot(dp, dv) / dvdv)
                else:
                    t_cpa = 0.0
                feat[i, 21] = t_cpa                               # t_cpa
            feat[:n_uav, 22] = state.uav_avoiding.astype(np.float32)  # f_avoid

        offset = n_uav
        for i in range(n_sec):
            sec = state.sector_occupancy[i]
            end = min(len(sec), self.feature_dim)
            feat[offset + i, :end] = sec[:end]

        offset += n_sec
        for i in range(n_wx):
            wx = state.weather_cells[i]
            end = min(len(wx), self.feature_dim)
            feat[offset + i, :end] = wx[:end]

        offset += n_wx
        for i in range(n_rz):
            rz = state.restricted_zones[i]
            end = min(len(rz), self.feature_dim)
            feat[offset + i, :end] = rz[:end]

        return feat

    def _build_node_types(
        self, n_uav: int, n_sec: int, n_wx: int, n_rz: int
    ) -> np.ndarray:
        types = np.concatenate([
            np.full(n_uav, ENTITY_TYPES["uav"]),
            np.full(n_sec, ENTITY_TYPES["sector"]),
            np.full(n_wx, ENTITY_TYPES["weather"]),
            np.full(n_rz, ENTITY_TYPES["restricted"]),
        ])
        return types

    def _build_edges(
        self,
        state: AirspaceState,
        n_uav: int,
        n_sec: int,
        n_wx: int,
        n_rz: int,
    ) -> Tuple[Dict[int, Tuple[List, List]], Dict[int, List]]:
        edge_indices: Dict[int, Tuple[List, List]] = {r: ([], []) for r in range(6)}
        edge_deltas: Dict[int, List] = {r: [] for r in range(6)}
        t = state.epoch_time

        self._add_approach_edges(state, n_uav, t, edge_indices, edge_deltas)
        self._add_corridor_edges(state, n_uav, t, edge_indices, edge_deltas)
        self._add_weather_edges(state, n_uav, n_sec, n_wx, t, edge_indices, edge_deltas)
        self._add_restriction_edges(state, n_uav, n_sec, n_wx, n_rz, t, edge_indices, edge_deltas)

        return edge_indices, edge_deltas

    def _add_approach_edges(self, state, n_uav, t, edge_indices, edge_deltas):
        r_approach = RELATION_VOCAB["approaches"]
        r_conflict = RELATION_VOCAB["conflicts_with"]

        for i in range(n_uav):
            for j in range(i + 1, n_uav):
                dp = state.uav_positions[j] - state.uav_positions[i]
                dv = state.uav_velocities[j] - state.uav_velocities[i]

                h_dist = np.sqrt(dp[0] ** 2 + dp[1] ** 2)
                v_dist = abs(dp[2])

                speed_close = np.dot(dp[:2], dv[:2])
                is_closing = speed_close < 0

                cpa_h = h_dist
                if is_closing and np.dot(dv, dv) > 1e-8:
                    t_cpa = -np.dot(dp, dv) / np.dot(dv, dv)
                    t_cpa = np.clip(t_cpa, 0, self.approach_lookahead)
                    cpa_pos = dp + dv * t_cpa
                    cpa_h = np.sqrt(cpa_pos[0] ** 2 + cpa_pos[1] ** 2)

                if cpa_h < self.approach_cpa_h and v_dist < self.approach_cpa_v:
                    delta = t - self._last_edge_times[r_approach].get((i, j), t)
                    self._last_edge_times[r_approach][(i, j)] = t

                    for src, dst in [(i, j), (j, i)]:
                        edge_indices[r_approach][0].append(src)
                        edge_indices[r_approach][1].append(dst)
                        edge_deltas[r_approach].append(delta)

                    if cpa_h < self.approach_cpa_h * 0.3:
                        delta_c = t - self._last_edge_times[r_conflict].get((i, j), t)
                        self._last_edge_times[r_conflict][(i, j)] = t
                        for src, dst in [(i, j), (j, i)]:
                            edge_indices[r_conflict][0].append(src)
                            edge_indices[r_conflict][1].append(dst)
                            edge_deltas[r_conflict].append(delta_c)

    def _add_corridor_edges(self, state, n_uav, t, edge_indices, edge_deltas):
        r = RELATION_VOCAB["shares_corridor"]
        corridor_map: Dict[int, List[int]] = {}

        for uav_a, uav_b, start_t, end_t in state.corridor_reservations:
            if t <= end_t and (t + self.corridor_lookahead) >= start_t:
                seg_id = hash((min(uav_a, uav_b), max(uav_a, uav_b)))
                corridor_map.setdefault(seg_id, [])
                if uav_a < n_uav:
                    corridor_map[seg_id].append(uav_a)
                if uav_b < n_uav:
                    corridor_map[seg_id].append(uav_b)

        for seg_uavs in corridor_map.values():
            uavs = list(set(seg_uavs))
            for a in range(len(uavs)):
                for b in range(a + 1, len(uavs)):
                    i, j = uavs[a], uavs[b]
                    delta = t - self._last_edge_times[r].get((i, j), t)
                    self._last_edge_times[r][(i, j)] = t
                    for src, dst in [(i, j), (j, i)]:
                        edge_indices[r][0].append(src)
                        edge_indices[r][1].append(dst)
                        edge_deltas[r].append(delta)

    def _add_weather_edges(self, state, n_uav, n_sec, n_wx, t, edge_indices, edge_deltas):
        r_wind = RELATION_VOCAB["is_downwind_of"]
        wx_offset = n_uav + n_sec

        if n_wx == 0:
            return

        wx_positions = state.weather_cells[:, :3] if state.weather_cells.shape[1] >= 3 else None
        if wx_positions is None:
            return

        for i in range(n_uav):
            for w in range(n_wx):
                d = np.linalg.norm(state.uav_positions[i, :2] - wx_positions[w, :2])
                if d < self.weather_radius:
                    wx_node = wx_offset + w
                    delta = t - self._last_edge_times[r_wind].get((i, wx_node), t)
                    self._last_edge_times[r_wind][(i, wx_node)] = t
                    edge_indices[r_wind][0].append(wx_node)
                    edge_indices[r_wind][1].append(i)
                    edge_deltas[r_wind].append(delta)

    def _add_restriction_edges(self, state, n_uav, n_sec, n_wx, n_rz, t, edge_indices, edge_deltas):
        r = RELATION_VOCAB["is_restricted_by"]
        rz_offset = n_uav + n_sec + n_wx

        for i in range(n_uav):
            for z in range(n_rz):
                rz_center = state.restricted_zones[z, :3]
                rz_radius = state.restricted_zones[z, 3] if state.restricted_zones.shape[1] > 3 else 200.0
                d = np.linalg.norm(state.uav_positions[i, :2] - rz_center[:2])
                if d < rz_radius * 1.5:
                    rz_node = rz_offset + z
                    delta = t - self._last_edge_times[r].get((i, rz_node), t)
                    self._last_edge_times[r][(i, rz_node)] = t
                    edge_indices[r][0].append(rz_node)
                    edge_indices[r][1].append(i)
                    edge_deltas[r].append(delta)

    def reset(self):
        """Clear cached edge timestamps between scenarios."""
        self._last_edge_times = {r: {} for r in range(6)}

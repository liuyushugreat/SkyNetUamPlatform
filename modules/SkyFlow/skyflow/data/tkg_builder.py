"""Temporal Knowledge Graph construction from multi-source airspace telemetry.

Constructs typed entity-relation-time graphs from four parallel data
streams: ADS-B telemetry, flight plans, weather grid, and corridor
reservation log. Designed to execute in < 12 ms on benchmark hardware.
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
        feat = np.zeros((n_total, self.feature_dim), dtype=np.float32)

        for i in range(n_uav):
            feat[i, 0:3] = state.uav_positions[i]
            feat[i, 3:6] = state.uav_velocities[i]
            feat[i, 6] = state.uav_headings[i]
            feat[i, 7] = state.uav_battery[i]
            feat[i, 8] = state.uav_priority[i]
            feat[i, 9] = float(state.uav_avoiding[i])
            speed = np.linalg.norm(state.uav_velocities[i])
            feat[i, 10] = speed
            feat[i, 11] = state.uav_positions[i][2] / 200.0
            feat[i, 12:15] = state.uav_velocities[i] / (speed + 1e-8)

        offset = n_uav
        for i in range(n_sec):
            sec = state.sector_occupancy[i]
            end = min(len(sec), self.feature_dim - 15)
            feat[offset + i, 15 : 15 + end] = sec[:end]

        offset += n_sec
        for i in range(n_wx):
            wx = state.weather_cells[i]
            end = min(len(wx), self.feature_dim - 15)
            feat[offset + i, 15 : 15 + end] = wx[:end]

        offset += n_wx
        for i in range(n_rz):
            rz = state.restricted_zones[i]
            end = min(len(rz), self.feature_dim - 15)
            feat[offset + i, 15 : 15 + end] = rz[:end]

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

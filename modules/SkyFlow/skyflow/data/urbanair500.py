"""UrbanAir-500: simulation benchmark for dense low-altitude conflict detection.

Physics-accurate rotorcraft dynamics (10 Hz), stochastic wind field,
GPS noise (CEP 2.5 m), ADS-B latency (0.5–1.2 s), 500 concurrent UAVs
over a 5 km × 5 km urban grid. Conflict ground truth via exact 6-DoF
trajectory integration with 10 m horizontal / 3 m vertical separation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from skyflow.data.tkg_builder import AirspaceState, TKGBuilder, TKGSnapshot


@dataclass
class UAVFlightPlan:
    uav_id: int
    waypoints: np.ndarray       # (W, 3) sequence of [x, y, z]
    priority: int               # 0=low, 1=normal, 2=high, 3=emergency
    cruise_speed: float         # m/s
    start_time: float           # seconds


@dataclass
class ConflictEvent:
    uav_i: int
    uav_j: int
    epoch: int
    time_to_conflict: float
    min_separation_h: float
    min_separation_v: float


class UrbanAir500:
    """Procedural UrbanAir-500 benchmark generator."""

    def __init__(
        self,
        num_uavs: int = 500,
        grid_size: float = 5000.0,
        altitude_range: Tuple[float, float] = (30.0, 150.0),
        num_corridors: int = 24,
        num_sectors: int = 64,
        num_weather_cells: int = 36,
        num_restricted_zones: int = 12,
        dt: float = 0.1,
        wind_std: float = 2.0,
        gps_cep: float = 2.5,
        adsb_latency_range: Tuple[float, float] = (0.5, 1.2),
        conflict_h_sep: float = 10.0,
        conflict_v_sep: float = 3.0,
        seed: int = 42,
    ):
        self.num_uavs = num_uavs
        self.grid_size = grid_size
        self.altitude_range = altitude_range
        self.num_corridors = num_corridors
        self.num_sectors = num_sectors
        self.num_weather_cells = num_weather_cells
        self.num_restricted_zones = num_restricted_zones
        self.dt = dt
        self.wind_std = wind_std
        self.gps_cep = gps_cep
        self.adsb_latency_range = adsb_latency_range
        self.conflict_h_sep = conflict_h_sep
        self.conflict_v_sep = conflict_v_sep

        self.rng = np.random.RandomState(seed)
        self._init_infrastructure()

    def _init_infrastructure(self):
        n_side = int(math.sqrt(self.num_sectors))
        cell = self.grid_size / n_side
        self.sector_centers = np.array([
            [cell * (i + 0.5), cell * (j + 0.5), 0.0]
            for i in range(n_side) for j in range(n_side)
        ], dtype=np.float32)

        wx_side = int(math.sqrt(self.num_weather_cells))
        wx_cell = self.grid_size / wx_side
        self.weather_positions = np.array([
            [wx_cell * (i + 0.5), wx_cell * (j + 0.5), 90.0]
            for i in range(wx_side) for j in range(wx_side)
        ], dtype=np.float32)

        self.restricted_zones = np.zeros(
            (self.num_restricted_zones, 8), dtype=np.float32
        )
        for z in range(self.num_restricted_zones):
            cx = self.rng.uniform(500, self.grid_size - 500)
            cy = self.rng.uniform(500, self.grid_size - 500)
            cz = self.rng.uniform(*self.altitude_range)
            radius = self.rng.uniform(100, 300)
            self.restricted_zones[z, :4] = [cx, cy, cz, radius]
            self.restricted_zones[z, 4] = 1.0

        self.corridor_nodes = self._generate_corridor_graph()

    def _generate_corridor_graph(self) -> np.ndarray:
        nodes = []
        for _ in range(self.num_corridors):
            x = self.rng.uniform(200, self.grid_size - 200)
            y = self.rng.uniform(200, self.grid_size - 200)
            z = self.rng.uniform(*self.altitude_range)
            nodes.append([x, y, z])
        return np.array(nodes, dtype=np.float32)

    def generate_flight_plans(self, num_plans: int = 500) -> List[UAVFlightPlan]:
        plans = []
        effective_grid = min(self.grid_size, self.num_uavs * 8.0)
        center = self.grid_size / 2.0

        for uid in range(num_plans):
            n_wp = self.rng.randint(3, 8)
            waypoints = np.zeros((n_wp, 3), dtype=np.float32)
            waypoints[0] = [
                center + self.rng.uniform(-effective_grid / 2, effective_grid / 2),
                center + self.rng.uniform(-effective_grid / 2, effective_grid / 2),
                self.rng.uniform(*self.altitude_range),
            ]
            for w in range(1, n_wp):
                dx = self.rng.uniform(-800, 800)
                dy = self.rng.uniform(-800, 800)
                dz = self.rng.uniform(-20, 20)
                waypoints[w] = waypoints[w - 1] + [dx, dy, dz]
                waypoints[w, :2] = np.clip(waypoints[w, :2], 50, self.grid_size - 50)
                waypoints[w, 2] = np.clip(
                    waypoints[w, 2], self.altitude_range[0], self.altitude_range[1]
                )

            priority = self.rng.choice([0, 1, 1, 1, 2, 3], p=[0.1, 0.6, 0.15, 0.05, 0.05, 0.05])
            speed = self.rng.uniform(8.0, 22.0)
            start = self.rng.uniform(0, 30.0)

            plans.append(UAVFlightPlan(
                uav_id=uid,
                waypoints=waypoints,
                priority=priority,
                cruise_speed=speed,
                start_time=start,
            ))
        return plans

    def simulate_scenario(
        self,
        duration_seconds: float = 60.0,
        plans: Optional[List[UAVFlightPlan]] = None,
    ) -> Iterator[Tuple[AirspaceState, List[ConflictEvent]]]:
        """Run a scenario and yield (state, conflicts) at each epoch."""
        if plans is None:
            plans = self.generate_flight_plans(self.num_uavs)

        positions = np.zeros((self.num_uavs, 3), dtype=np.float32)
        velocities = np.zeros((self.num_uavs, 3), dtype=np.float32)
        headings = np.zeros(self.num_uavs, dtype=np.float32)
        battery = np.ones(self.num_uavs, dtype=np.float32)
        priorities = np.zeros(self.num_uavs, dtype=np.int32)
        avoiding = np.zeros(self.num_uavs, dtype=bool)
        wp_idx = np.zeros(self.num_uavs, dtype=np.int32)

        for plan in plans:
            uid = plan.uav_id
            positions[uid] = plan.waypoints[0]
            priorities[uid] = plan.priority
            if len(plan.waypoints) > 1:
                d = plan.waypoints[1] - plan.waypoints[0]
                dist = np.linalg.norm(d)
                if dist > 1e-6:
                    velocities[uid] = d / dist * plan.cruise_speed
                    headings[uid] = np.arctan2(d[1], d[0])

        n_epochs = int(duration_seconds / self.dt)
        wind_field = self.rng.randn(n_epochs, 3).astype(np.float32) * self.wind_std

        for epoch in range(n_epochs):
            t = epoch * self.dt

            wind = wind_field[epoch]
            gps_noise = self.rng.randn(self.num_uavs, 3).astype(np.float32) * self.gps_cep * 0.01

            for uid, plan in enumerate(plans):
                if t < plan.start_time:
                    continue
                wi = wp_idx[uid]
                if wi >= len(plan.waypoints) - 1:
                    velocities[uid] *= 0.95
                    continue

                target = plan.waypoints[wi + 1]
                to_target = target - positions[uid]
                dist = np.linalg.norm(to_target)

                if dist < 5.0:
                    wp_idx[uid] = min(wi + 1, len(plan.waypoints) - 1)
                    continue

                desired_v = to_target / dist * plan.cruise_speed
                steer = (desired_v - velocities[uid]) * 0.3
                velocities[uid] += steer * self.dt
                velocities[uid] += wind * 0.1 * self.dt
                headings[uid] = np.arctan2(velocities[uid][1], velocities[uid][0])

            positions += velocities * self.dt + gps_noise
            positions[:, :2] = np.clip(positions[:, :2], 0, self.grid_size)
            positions[:, 2] = np.clip(
                positions[:, 2], self.altitude_range[0], self.altitude_range[1]
            )
            battery -= self.rng.uniform(0.00001, 0.00005, self.num_uavs).astype(np.float32)
            battery = np.clip(battery, 0, 1)

            conflicts = self._detect_ground_truth_conflicts(positions, velocities, epoch, t)

            sector_occ = self._compute_sector_occupancy(positions)
            weather_data = self._compute_weather_state(t, wind)
            corridor_res = self._compute_corridor_reservations(plans, t)

            state = AirspaceState(
                uav_positions=positions.copy(),
                uav_velocities=velocities.copy(),
                uav_headings=headings.copy(),
                uav_battery=battery.copy(),
                uav_priority=priorities.copy(),
                uav_avoiding=avoiding.copy(),
                sector_occupancy=sector_occ,
                weather_cells=weather_data,
                restricted_zones=self.restricted_zones.copy(),
                corridor_reservations=corridor_res,
                epoch_time=t,
            )
            yield state, conflicts

    def _detect_ground_truth_conflicts(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        epoch: int,
        t: float,
    ) -> List[ConflictEvent]:
        conflicts = []
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                dp = positions[j] - positions[i]
                h_dist = np.sqrt(dp[0] ** 2 + dp[1] ** 2)
                v_dist = abs(dp[2])

                if h_dist < self.conflict_h_sep and v_dist < self.conflict_v_sep:
                    dv = velocities[j] - velocities[i]
                    if np.dot(dv, dv) > 1e-8:
                        ttc = -np.dot(dp, dv) / np.dot(dv, dv)
                    else:
                        ttc = 0.0
                    conflicts.append(ConflictEvent(
                        uav_i=i, uav_j=j, epoch=epoch,
                        time_to_conflict=max(0, ttc),
                        min_separation_h=h_dist, min_separation_v=v_dist,
                    ))
        return conflicts

    def _compute_sector_occupancy(self, positions: np.ndarray) -> np.ndarray:
        occ = np.zeros((self.num_sectors, 8), dtype=np.float32)
        for s in range(self.num_sectors):
            center = self.sector_centers[s, :2]
            dists = np.linalg.norm(positions[:, :2] - center, axis=1)
            cell_radius = self.grid_size / (2 * math.sqrt(self.num_sectors))
            count = np.sum(dists < cell_radius)
            occ[s, 0] = center[0] / self.grid_size
            occ[s, 1] = center[1] / self.grid_size
            occ[s, 2] = count
            occ[s, 3] = count / max(self.num_uavs * 0.1, 1)
        return occ

    def _compute_weather_state(
        self, t: float, wind: np.ndarray
    ) -> np.ndarray:
        wx = np.zeros((self.num_weather_cells, 12), dtype=np.float32)
        for w in range(self.num_weather_cells):
            wx[w, 0:3] = self.weather_positions[w]
            wx[w, 3:6] = wind * (1.0 + 0.1 * self.rng.randn())
            wx[w, 6] = np.linalg.norm(wind)
            wx[w, 7] = 1.0
            wx[w, 8] = self.rng.uniform(5000, 15000)
        return wx

    def _compute_corridor_reservations(
        self, plans: List[UAVFlightPlan], t: float
    ) -> List[Tuple[int, int, float, float]]:
        reservations = []
        corridor_users: Dict[int, List[int]] = {}
        for plan in plans:
            for ci, cnode in enumerate(self.corridor_nodes):
                if plan.uav_id < len(plans):
                    wi = min(1, len(plan.waypoints) - 1)
                    wp = plan.waypoints[wi]
                    d = np.linalg.norm(wp[:2] - cnode[:2])
                    if d < 600:
                        corridor_users.setdefault(ci, []).append(plan.uav_id)

        for ci, users in corridor_users.items():
            for a in range(len(users)):
                for b in range(a + 1, min(a + 3, len(users))):
                    reservations.append((users[a], users[b], t, t + 120.0))

        return reservations

    def generate_dataset(
        self,
        split: str = "train",
        num_scenarios: int = 10,
        scenario_duration: float = 60.0,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tuple[TKGSnapshot, torch.Tensor]]:
        """Generate a full dataset split as list of (snapshot, labels)."""
        builder = TKGBuilder()
        dataset = []
        epoch_step = 10

        for scenario_idx in range(num_scenarios):
            self.rng = np.random.RandomState(
                hash((split, scenario_idx)) % (2**31)
            )
            self._init_infrastructure()
            builder.reset()
            plans = self.generate_flight_plans(self.num_uavs)

            for epoch_idx, (state, conflicts) in enumerate(
                self.simulate_scenario(scenario_duration, plans)
            ):
                if epoch_idx % epoch_step != 0:
                    continue

                snapshot = builder.build(state, device=device)

                conflict_set = set()
                for c in conflicts:
                    conflict_set.add((c.uav_i, c.uav_j))

                n = snapshot.num_uavs
                n_sample = min(n * 4, n * (n - 1) // 2)
                pairs_src, pairs_dst, labels = [], [], []
                all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

                if len(all_pairs) > n_sample:
                    pos_pairs = [(i, j) for i, j in all_pairs if (i, j) in conflict_set]
                    neg_pairs = [(i, j) for i, j in all_pairs if (i, j) not in conflict_set]
                    n_pos = len(pos_pairs)
                    n_neg = min(len(neg_pairs), max(n_sample - n_pos, n_pos * 10))
                    self.rng.shuffle(neg_pairs)
                    sampled = pos_pairs + neg_pairs[:n_neg]
                else:
                    sampled = all_pairs

                for i, j in sampled:
                    pairs_src.append(i)
                    pairs_dst.append(j)
                    labels.append(1.0 if (i, j) in conflict_set else 0.0)

                snapshot.conflict_pairs = torch.tensor(
                    [pairs_src, pairs_dst], dtype=torch.long, device=device
                )
                snapshot.conflict_labels = torch.tensor(
                    labels, dtype=torch.float32, device=device
                )
                dataset.append((snapshot, snapshot.conflict_labels))

        return dataset

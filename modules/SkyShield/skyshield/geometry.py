"""Lightweight geometry helpers (planar approximation in metres).

The defended urban region is a square that covers ``area_km2`` square
kilometres, centred at the origin.  Radars are placed deterministically
on a uniform grid; targets are sampled inside the square from a
seeded RNG so that the same scenario regenerates byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class Point:
    x: float  # metres east of origin
    y: float  # metres north of origin
    z: float = 100.0  # altitude above ground (metres)

    def horizontal_distance(self, other: "Point") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def slant_distance(self, other: "Point") -> float:
        return sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


def square_side_m(area_km2: float) -> float:
    return float(sqrt(area_km2)) * 1000.0


def radar_grid(num_nodes: int, area_km2: float) -> list[Point]:
    """Place ``num_nodes`` radars on a corner / inscribed pattern.

    1 -> centre; 2 -> opposite corners; 4 -> all corners; >=6 -> ring.
    """
    side = square_side_m(area_km2)
    half = side * 0.5
    if num_nodes <= 0:
        return []
    if num_nodes == 1:
        return [Point(0.0, 0.0)]
    if num_nodes == 2:
        return [Point(-half * 0.5, -half * 0.5), Point(half * 0.5, half * 0.5)]
    if num_nodes == 4:
        return [
            Point(-half * 0.5, -half * 0.5),
            Point(half * 0.5, -half * 0.5),
            Point(-half * 0.5, half * 0.5),
            Point(half * 0.5, half * 0.5),
        ]
    # >= 5: ring at radius = 0.5 * half_side
    radius = half * 0.55
    pts = []
    for k in range(num_nodes):
        ang = 2.0 * np.pi * k / num_nodes
        pts.append(Point(radius * np.cos(ang), radius * np.sin(ang)))
    return pts


def random_target_track(
    rng: np.random.Generator,
    area_km2: float,
    speed_mps: float,
    altitude_m: float,
    duration_s: float,
    dt_s: float = 0.1,
    maneuver_g: float = 0.5,
) -> np.ndarray:
    """Generate a synthetic target track with bounded lateral acceleration.

    Returns an ``(N, 3)`` array of (x, y, z) samples in metres.  The
    ``maneuver_g`` parameter scales lateral acceleration noise that
    perturbs the heading at every step, mimicking a target that
    occasionally banks (this is the main stressor in E3).
    """
    side = square_side_m(area_km2)
    half = side * 0.5
    # Spawn on a random edge, fly across the square.
    edge = rng.integers(0, 4)
    if edge == 0:
        start = np.array([-half, rng.uniform(-half, half)])
        heading = 0.0
    elif edge == 1:
        start = np.array([half, rng.uniform(-half, half)])
        heading = np.pi
    elif edge == 2:
        start = np.array([rng.uniform(-half, half), -half])
        heading = np.pi / 2
    else:
        start = np.array([rng.uniform(-half, half), half])
        heading = -np.pi / 2

    n_steps = int(duration_s / dt_s)
    pos = start.astype(np.float64).copy()
    out = np.zeros((n_steps, 3), dtype=np.float64)
    out[:, 2] = altitude_m
    g = 9.81
    accel_lat_max = maneuver_g * g
    for i in range(n_steps):
        # heading random walk with bounded variance
        omega = rng.normal(0.0, accel_lat_max / max(1.0, speed_mps)) * dt_s
        heading += omega
        pos[0] += speed_mps * np.cos(heading) * dt_s
        pos[1] += speed_mps * np.sin(heading) * dt_s
        # clamp to box (reflect)
        if abs(pos[0]) > half:
            pos[0] = np.clip(pos[0], -half, half)
            heading = np.pi - heading
        if abs(pos[1]) > half:
            pos[1] = np.clip(pos[1], -half, half)
            heading = -heading
        out[i, 0] = pos[0]
        out[i, 1] = pos[1]
    return out

"""Mobility models.

Three models are supported, all returning *windowed* positions for every
entity at each second:

* ``corridor``       — entities travel roughly along one of K corridors
                        (typical of UAM route structures); noise is
                        Gaussian around the corridor centerline.
* ``random_waypoint``— every ``T_wp`` s each entity picks a new uniform
                        destination and moves at constant speed.
* ``levy``           — Lévy-flight model; heavy-tailed step lengths,
                        captures bursty movement.

All models are seeded and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MobilityConfig, WorkloadConfig


@dataclass
class PositionTrack:
    """Per-entity x/y positions sampled at 1 Hz inside the square area."""
    xs: np.ndarray   # shape (T+1, N)
    ys: np.ndarray   # shape (T+1, N)

    @property
    def num_entities(self) -> int:
        return int(self.xs.shape[1])

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        idx = int(np.clip(np.floor(t), 0, self.xs.shape[0] - 1))
        return self.xs[idx], self.ys[idx]


def build_corridor_tracks(
    workload: WorkloadConfig,
    rng: np.random.Generator,
) -> PositionTrack:
    """Eight-corridor model tiled over the square area.

    Corridors are drawn as axis-aligned and two diagonals; each entity is
    assigned to one corridor, starts at a uniformly-random position along
    it, and jitters with Gaussian noise proportional to ``corridor_sigma``.
    """
    L = workload.area_km
    T = int(round(workload.duration_s)) + 1
    N = workload.num_entities
    speed_kmps = (workload.mobility.speed_mps / 1000.0)
    sigma = workload.mobility.corridor_sigma * L

    corridors = [
        (np.array([0.0, 0.3 * L]), np.array([1.0,  0.0])),   # east
        (np.array([0.0, 0.7 * L]), np.array([1.0,  0.0])),   # east
        (np.array([0.5 * L, 0.0]), np.array([0.0,  1.0])),   # north
        (np.array([0.5 * L, L   ]), np.array([0.0, -1.0])),  # south
        (np.array([0.0, 0.0]),    np.array([1.0,  1.0]) / np.sqrt(2)),   # NE diag
        (np.array([L, 0.0]),      np.array([-1.0, 1.0]) / np.sqrt(2)),   # NW diag
        (np.array([0.0, L]),      np.array([1.0, -1.0]) / np.sqrt(2)),   # SE diag
        (np.array([L, L]),        np.array([-1.0,-1.0]) / np.sqrt(2)),   # SW diag
    ]

    assign = rng.integers(0, len(corridors), size=N)
    phase = rng.uniform(0.0, L, size=N)
    direction_sign = np.where(rng.random(N) < 0.5, 1.0, -1.0)

    xs = np.zeros((T, N), dtype=np.float32)
    ys = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        for k, (origin, direction) in enumerate(corridors):
            mask = assign == k
            if not np.any(mask):
                continue
            s = (phase[mask] + direction_sign[mask] * speed_kmps * t) % L
            px = origin[0] + direction[0] * s
            py = origin[1] + direction[1] * s
            nx = rng.normal(0.0, sigma, size=int(mask.sum()))
            ny = rng.normal(0.0, sigma, size=int(mask.sum()))
            xs[t, mask] = np.clip(px + nx, 0.0, L)
            ys[t, mask] = np.clip(py + ny, 0.0, L)
    return PositionTrack(xs=xs, ys=ys)


def build_random_waypoint_tracks(
    workload: WorkloadConfig,
    rng: np.random.Generator,
) -> PositionTrack:
    L = workload.area_km
    T = int(round(workload.duration_s)) + 1
    N = workload.num_entities
    speed = workload.mobility.speed_mps / 1000.0
    xs = np.zeros((T, N), dtype=np.float32)
    ys = np.zeros((T, N), dtype=np.float32)
    px = rng.uniform(0.0, L, size=N)
    py = rng.uniform(0.0, L, size=N)
    tx = rng.uniform(0.0, L, size=N)
    ty = rng.uniform(0.0, L, size=N)
    for t in range(T):
        dx = tx - px
        dy = ty - py
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        step = np.minimum(dist, speed)
        px = px + step * dx / dist
        py = py + step * dy / dist
        arrived = dist <= speed
        if np.any(arrived):
            tx[arrived] = rng.uniform(0.0, L, size=int(arrived.sum()))
            ty[arrived] = rng.uniform(0.0, L, size=int(arrived.sum()))
        xs[t] = px
        ys[t] = py
    return PositionTrack(xs=xs, ys=ys)


def build_levy_tracks(
    workload: WorkloadConfig,
    rng: np.random.Generator,
) -> PositionTrack:
    L = workload.area_km
    T = int(round(workload.duration_s)) + 1
    N = workload.num_entities
    speed = workload.mobility.speed_mps / 1000.0
    xs = np.zeros((T, N), dtype=np.float32)
    ys = np.zeros((T, N), dtype=np.float32)
    px = rng.uniform(0.0, L, size=N)
    py = rng.uniform(0.0, L, size=N)
    mu = 1.5
    for t in range(T):
        # Lévy step length in km (heavy-tailed Pareto-ish).
        r = speed * (1.0 + rng.pareto(mu, size=N))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
        px = np.clip(px + r * np.cos(theta), 0.0, L)
        py = np.clip(py + r * np.sin(theta), 0.0, L)
        xs[t] = px
        ys[t] = py
    return PositionTrack(xs=xs, ys=ys)


def build_tracks(
    workload: WorkloadConfig,
    rng: np.random.Generator,
) -> PositionTrack:
    model = workload.mobility.model
    if model == "corridor":
        return build_corridor_tracks(workload, rng)
    if model == "random_waypoint":
        return build_random_waypoint_tracks(workload, rng)
    if model == "levy":
        return build_levy_tracks(workload, rng)
    raise ValueError(f"unknown mobility model: {model}")

"""Deterministic city-scale UAM workload.

A workload is a stream of ``Event`` objects, one per (entity, emission
time) pair.  The generator is 100% seeded and *never* mutated after
construction, so every baseline sees bit-identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..config import WorkloadConfig
from . import mobility
from .dag import TaskDAG


@dataclass
class Entity:
    eid: int
    home_cell: int
    x0: float
    y0: float


@dataclass
class Event:
    event_id: int
    eid: int
    t: float                    # emission time (s, virtual)
    x: float
    y: float
    rule_fan_in: int            # # symbolic rules that will read this event
    jitter_multiplier: float    # local burst multiplier (already applied to rate)
    # Ids of spatially-nearby entities that ``rule_check`` will need to
    # consult (e.g. other aircraft whose trajectories must not conflict).
    # The length is ``rule_fan_in - 1``; may be empty for singleton rules.
    peer_eids: tuple[int, ...] = ()
    features: np.ndarray = field(default_factory=lambda: np.zeros(0))


class CityScaleWorkload:
    """Reproducible city-scale workload (entities + event stream + DAG).

    The generator is one-shot: call :py:meth:`events` once and iterate.
    Calling it twice with the same seed yields an identical stream.
    """

    def __init__(
        self,
        cfg: WorkloadConfig,
        dag: TaskDAG,
        seed: int,
        cells_per_side: int = 16,
    ) -> None:
        self.cfg = cfg
        self.dag = dag
        self.seed = int(seed)
        self.cells_per_side = int(cells_per_side)
        self._rng = np.random.default_rng(self.seed)
        self.tracks = mobility.build_tracks(cfg, self._rng)
        self.entities = self._build_entities()

    # ---------------------------------------------------------------- internals

    def _build_entities(self) -> list[Entity]:
        L = self.cfg.area_km
        C = self.cells_per_side
        xs0 = self.tracks.xs[0]
        ys0 = self.tracks.ys[0]
        cells = (np.clip((ys0 / L * C).astype(np.int32), 0, C - 1) * C
                 + np.clip((xs0 / L * C).astype(np.int32), 0, C - 1))
        return [
            Entity(eid=i, home_cell=int(cells[i]),
                   x0=float(xs0[i]), y0=float(ys0[i]))
            for i in range(self.cfg.num_entities)
        ]

    def _jitter_profile(self) -> np.ndarray:
        """Multiplicative jitter per 5-second window, CoV=cfg.jitter.cov."""
        T = int(round(self.cfg.duration_s)) + 1
        n_wind = max(1, T // 5)
        cov = self.cfg.jitter.cov
        w = self._rng.normal(1.0, cov, size=n_wind).clip(0.1, 5.0)
        return np.repeat(w, 5)[:T]

    # ---------------------------------------------------------------- public

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_leaf_ops(self) -> int:
        return len(self.dag.leaves())

    def events(self) -> Iterator[Event]:
        """Iterate over the full event stream in strict emission-time order.

        Within each 1s window we sort all (entity, t-offset) pairs by
        t-offset so the runtime sees a monotonically-increasing arrival
        timestamp.  This is required for the DES queue model to produce
        meaningful latencies (out-of-order arrivals would artificially
        inflate the effective queueing time).
        """
        cfg = self.cfg
        rng = self._rng
        rate = cfg.event_rate_per_entity_s
        duration = cfg.duration_s
        jitter = self._jitter_profile()

        eid_arr = np.arange(len(self.entities))

        T = int(np.ceil(duration))
        counts = np.zeros((T, len(self.entities)), dtype=np.int32)
        for t in range(T):
            counts[t] = rng.poisson(lam=rate * jitter[t % len(jitter)],
                                    size=len(self.entities))

        # Precompute a fast spatial-neighbour index per 1-second frame so
        # that peer lookups reflect the *physical* proximity of entities.
        # We bucket entities into a grid with cell size ≈ area / (C^2) and,
        # for each emitting entity, sample its peers uniformly from its
        # bucket and the 8 surrounding buckets.  This makes STP
        # (spatially-aware) outperform random / hash partitioners
        # whenever rule_fan_in > 1.
        C = self.cells_per_side
        L = cfg.area_km

        ev_id = 0
        for t in range(T):
            layer_counts = counts[t]
            total = int(layer_counts.sum())
            if total == 0:
                continue
            xs, ys = self.tracks.at(float(t))
            # Rebuild the spatial index for this frame.
            cols = np.clip((xs / L * C).astype(np.int32), 0, C - 1)
            rows = np.clip((ys / L * C).astype(np.int32), 0, C - 1)
            cell_of_eid = rows * C + cols
            bucket: dict[int, list[int]] = {}
            for e in range(len(eid_arr)):
                bucket.setdefault(int(cell_of_eid[e]), []).append(e)

            # Expand per-entity counts into a flat array of entity ids.
            eids = np.repeat(eid_arr, layer_counts)
            offs = rng.uniform(0.0, 1.0, size=total).astype(np.float64)
            # Strict time ordering within the 1s window.
            order = np.argsort(offs, kind="stable")
            eids = eids[order]
            offs = offs[order]
            fan_ins = rng.integers(1, 4, size=total)
            jm = float(jitter[t % len(jitter)])
            for k in range(total):
                te = float(t) + float(offs[k])
                if te > duration:
                    continue
                e = int(eids[k])
                fi = int(fan_ins[k])
                peers: list[int] = []
                if fi > 1:
                    need = fi - 1
                    # Sample within a 3x3 neighbourhood of entity e's cell.
                    r0, c0 = int(rows[e]), int(cols[e])
                    candidates: list[int] = []
                    for dr in (-1, 0, 1):
                        rr = r0 + dr
                        if rr < 0 or rr >= C:
                            continue
                        for dc in (-1, 0, 1):
                            cc = c0 + dc
                            if cc < 0 or cc >= C:
                                continue
                            candidates.extend(bucket.get(rr * C + cc, ()))
                    if e in candidates:
                        candidates.remove(e)
                    if candidates:
                        pick = rng.choice(candidates,
                                          size=min(need, len(candidates)),
                                          replace=False)
                        peers = [int(p) for p in np.atleast_1d(pick)]
                yield Event(
                    event_id=ev_id,
                    eid=e,
                    t=te,
                    x=float(xs[e]),
                    y=float(ys[e]),
                    rule_fan_in=fi,
                    jitter_multiplier=jm,
                    peer_eids=tuple(peers),
                )
                ev_id += 1

    def replay(self) -> list[Event]:
        """Materialize the full stream (S/M scale only)."""
        return list(self.events())

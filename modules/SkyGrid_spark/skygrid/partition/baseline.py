"""Baseline partitioners used for comparison in the paper.

All partitioners share a small interface:

    ``assign(entities) -> Partition``

where ``entities`` is the list produced by
:py:class:`skygrid.workload.CityScaleWorkload`.  ``Partition`` then acts
as a pure dict-of-arrays carrying the edge assignment for every entity.

The baselines implemented here are the ones the paper reports against:

* **HashPartitioner**   — modulo-hash on entity id.
* **RandomPartitioner** — uniform random assignment (fixed seed).
* **LDGPartitioner**    — Linear Deterministic Greedy (Stanton & Kliot,
  KDD'12) streaming partitioner; picks the edge maximizing
  ``|N(v) ∩ P_i| * (1 - |P_i|/C)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from ..utils import deterministic_hash, make_rng


@dataclass
class Partition:
    num_edges: int
    assignment: np.ndarray          # shape (N,) int
    sizes: np.ndarray               # shape (num_edges,) int

    def of(self, eid: int) -> int:
        return int(self.assignment[eid])


class Partitioner(Protocol):
    num_edges: int

    def assign(self, entities) -> Partition: ...


# ---------------------------------------------------------------------- hash


class HashPartitioner:
    def __init__(self, num_edges: int) -> None:
        self.num_edges = int(num_edges)

    def assign(self, entities) -> Partition:
        N = len(entities)
        a = np.zeros(N, dtype=np.int32)
        for i, e in enumerate(entities):
            a[i] = deterministic_hash(f"ent-{e.eid}") % self.num_edges
        sizes = np.bincount(a, minlength=self.num_edges).astype(np.int32)
        return Partition(self.num_edges, a, sizes)


# ---------------------------------------------------------------------- random


class RandomPartitioner:
    def __init__(self, num_edges: int, seed: int = 12345) -> None:
        self.num_edges = int(num_edges)
        self.seed = int(seed)

    def assign(self, entities) -> Partition:
        N = len(entities)
        rng = make_rng(self.seed)
        a = rng.integers(0, self.num_edges, size=N).astype(np.int32)
        sizes = np.bincount(a, minlength=self.num_edges).astype(np.int32)
        return Partition(self.num_edges, a, sizes)


# ---------------------------------------------------------------------- LDG


class LDGPartitioner:
    """Streaming Linear Deterministic Greedy partitioner.

    We treat each entity's home cell as a small vertex neighborhood —
    entities sharing a cell are considered neighbors, which is a fair
    approximation of the entity-entity rule co-activation used by STP.
    """

    def __init__(
        self,
        num_edges: int,
        capacity_slack: float = 1.10,
    ) -> None:
        self.num_edges = int(num_edges)
        self.capacity_slack = float(capacity_slack)

    def assign(self, entities) -> Partition:
        N = len(entities)
        cap = int(np.ceil(self.capacity_slack * N / self.num_edges))
        sizes = np.zeros(self.num_edges, dtype=np.int32)
        cell_part_count = {}   # (cell, part) -> count
        a = np.full(N, -1, dtype=np.int32)
        order = np.argsort([e.eid for e in entities])
        for idx in order:
            e = entities[idx]
            best_score = -1.0
            best_p = 0
            for p in range(self.num_edges):
                if sizes[p] >= cap:
                    continue
                nbrs = cell_part_count.get((e.home_cell, p), 0)
                score = nbrs * (1.0 - sizes[p] / cap)
                if score > best_score:
                    best_score = score
                    best_p = p
            if best_score < 0:
                best_p = int(np.argmin(sizes))
            a[idx] = best_p
            sizes[best_p] += 1
            k = (e.home_cell, best_p)
            cell_part_count[k] = cell_part_count.get(k, 0) + 1
        return Partition(self.num_edges, a, sizes.astype(np.int32))

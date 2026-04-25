"""Spatio-Temporally aware Partitioner (STP) — paper contribution C1.

STP co-optimizes three objectives:

1. **Spatial compactness** — entities physically close (same spatial grid
   cell) should land on the same edge, so that edge-local rule operators
   can fire without cross-edge traffic.
2. **Rule-dependency co-location** — entities that trigger the same
   symbolic rules (approximated by ``rule_fan_in`` and cell co-occupancy)
   should be co-located to keep rule evaluation edge-local.
3. **Load balance** — per-edge load must stay within a ``(1 + γ)``
   multiplicative slack of the average, otherwise tail latency collapses.

Concretely, we solve a *guided cell-to-edge assignment* followed by two
rounds of FM-style boundary refinement.  The algorithm is linear in the
number of entities (``O(N + C · K)`` with ``C`` cells and ``K`` edges)
and is the streaming-friendly variant described in §5.1 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import make_rng
from .baseline import Partition


@dataclass
class STPParams:
    alpha_spatial: float = 1.0
    beta_ruledep: float = 0.6
    gamma_balance: float = 0.4
    refine_iters: int = 2
    cells_per_side: int = 16


class SpatioTemporalPartitioner:
    def __init__(
        self,
        num_edges: int,
        params: STPParams | None = None,
        seed: int = 20260928,
        capacity_weights: list[float] | None = None,
    ) -> None:
        self.num_edges = int(num_edges)
        self.params = params or STPParams()
        self.seed = int(seed)
        # Per-edge capacity weights (e.g. normalized TFLOPS).  When
        # ``None``, every edge is assigned an equal ``1/K`` share of
        # the global capacity.  A heterogeneous fabric (or a degraded
        # edge) is expressed as non-uniform weights, and STP then
        # matches cell mass to capacity so slow/failing edges receive
        # proportionally fewer entities.
        if capacity_weights is None:
            self.capacity_weights = np.full(self.num_edges, 1.0 / self.num_edges)
        else:
            w = np.asarray(capacity_weights, dtype=np.float64)
            w = w / max(1e-9, w.sum())
            self.capacity_weights = w

    # ------------------------------------------------------------------ API

    def assign(self, entities) -> Partition:
        N = len(entities)
        C = self.params.cells_per_side ** 2
        K = self.num_edges
        cw = self.capacity_weights

        cells = np.array([e.home_cell for e in entities], dtype=np.int32)
        cell_counts = np.bincount(cells, minlength=C)

        # Cell centroids in grid coordinates (row, col).
        side = self.params.cells_per_side
        rows = (np.arange(C) // side).astype(np.float64)
        cols = (np.arange(C) %  side).astype(np.float64)

        # Initial KMeans++-like seeding of K edge "centers" over populated cells.
        rng = make_rng(self.seed)
        populated = np.where(cell_counts > 0)[0]
        if populated.size == 0:
            return Partition(K, np.zeros(N, dtype=np.int32),
                             np.zeros(K, dtype=np.int32))

        centers = [int(rng.choice(populated))]
        while len(centers) < K:
            dists = np.full(C, np.inf)
            for c in centers:
                d = (rows - rows[c]) ** 2 + (cols - cols[c]) ** 2
                dists = np.minimum(dists, d)
            dists[cell_counts == 0] = 0.0
            if dists.sum() <= 0:
                centers.append(int(rng.choice(populated)))
                continue
            probs = dists / dists.sum()
            centers.append(int(rng.choice(C, p=probs)))

        center_rows = rows[centers]
        center_cols = cols[centers]

        # ---------------- cell → edge scoring
        # Per-edge capacity budget: cap_per_edge[k] = w[k] * N.  In the
        # homogeneous case w = 1/K for every edge and cap_per_edge
        # reduces to N/K.
        cap_per_edge = cw * float(N)
        cap_target_avg = float(np.mean(cap_per_edge))
        load = np.zeros(K, dtype=np.float64)
        cell_to_edge = np.full(C, -1, dtype=np.int32)

        # Hard capacity cap: partition k must not exceed (1 + γ) · cap_per_edge[k].
        hard_cap = (1.0 + self.params.gamma_balance) * cap_per_edge

        # Sort cells by density descending so the densest cells choose first.
        order = np.argsort(-cell_counts)
        for cell in order:
            if cell_counts[cell] == 0:
                continue
            # Spatial compactness: negative squared distance to center (log-normalized).
            sq = ((rows[cell] - center_rows) ** 2 +
                  (cols[cell] - center_cols) ** 2)
            spatial = -sq / max(1.0, side * side)

            # Rule dependency: reward co-location with already-assigned
            # neighbor cells on the same edge.
            ruledep = np.zeros(K, dtype=np.float64)
            r0, c0 = int(rows[cell]), int(cols[cell])
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < side and 0 <= nc < side:
                    nb = nr * side + nc
                    if cell_to_edge[nb] >= 0:
                        ruledep[cell_to_edge[nb]] += cell_counts[nb]
            ruledep = ruledep / max(1.0, float(cell_counts.max()))

            # Balance term: strictly negative for every partition whose
            # load exceeds its per-edge cap, scaled so that it can
            # outweigh the unit-magnitude spatial term when an edge is
            # materially over-budget.
            balance = -np.maximum(0.0, load - cap_per_edge) / max(1.0, cap_target_avg)

            # Hard constraint: mask out candidates that would overflow
            # their *per-edge* cap (so a degraded edge with low weight
            # reaches its cap early and is excluded).
            forbidden = (load + cell_counts[cell]) > hard_cap
            if forbidden.all():
                # Fall back to the least-loaded partition.
                choice = int(np.argmin(load))
            else:
                score = (self.params.alpha_spatial * spatial +
                         self.params.beta_ruledep * ruledep +
                         self.params.gamma_balance * balance)
                score[forbidden] = -1e18
                choice = int(np.argmax(score))
            cell_to_edge[cell] = choice
            load[choice] += cell_counts[cell]

        # ---------------- materialize entity assignment
        a = cell_to_edge[cells]
        sizes = np.bincount(a, minlength=K).astype(np.int32)

        # ---------------- FM-style boundary refinement
        for _ in range(self.params.refine_iters):
            a, sizes = _fm_refine(
                a, sizes, cells, cell_to_edge,
                side, cell_counts, self.params,
                num_edges=K, cap_per_edge=cap_per_edge,
            )

        return Partition(K, a.astype(np.int32), sizes.astype(np.int32))


# ---------------------------------------------------------------------- FM refinement


def _fm_refine(
    assignment: np.ndarray,
    sizes: np.ndarray,
    cells: np.ndarray,
    cell_to_edge: np.ndarray,
    side: int,
    cell_counts: np.ndarray,
    params: STPParams,
    *,
    num_edges: int,
    cap_per_edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Single FM pass: for each *boundary cell*, try moving it to an
    adjacent edge if doing so reduces the STP cost (spatial + ruledep +
    balance).  We bound the number of moves per pass to ``|boundary|``
    so a pass is O(C·K).
    """
    C = cell_to_edge.shape[0]
    # Identify boundary cells: any cell whose 4-neighbor is on a different edge.
    boundary: list[int] = []
    for cell in range(C):
        if cell_counts[cell] == 0:
            continue
        r0, c0 = cell // side, cell % side
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr < side and 0 <= nc < side:
                nb = nr * side + nc
                if cell_counts[nb] > 0 and cell_to_edge[nb] != cell_to_edge[cell]:
                    boundary.append(cell)
                    break

    for cell in boundary:
        cur = int(cell_to_edge[cell])
        r0, c0 = cell // side, cell % side
        # Candidate partitions: edges of 4-neighbors.
        candidates: set[int] = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr < side and 0 <= nc < side:
                nb = nr * side + nc
                if cell_counts[nb] > 0:
                    candidates.add(int(cell_to_edge[nb]))
        candidates.discard(cur)
        if not candidates:
            continue

        best_delta = 0.0
        best_p = cur
        for p in candidates:
            delta = _delta_cost(
                cell, cur, p, cell_to_edge, cell_counts,
                sizes, side, params, cap_per_edge,
            )
            if delta < best_delta:
                best_delta = delta
                best_p = p
        if best_p != cur and sizes[best_p] + cell_counts[cell] <= (
            (1.0 + params.gamma_balance) * cap_per_edge[best_p] + 0.5
        ):
            cell_to_edge[cell] = best_p
            moved = cell_counts[cell]
            sizes[cur] -= moved
            sizes[best_p] += moved

    assignment = cell_to_edge[cells]
    sizes = np.bincount(assignment, minlength=num_edges).astype(np.int32)
    return assignment, sizes


def _delta_cost(
    cell: int,
    cur: int,
    cand: int,
    cell_to_edge: np.ndarray,
    cell_counts: np.ndarray,
    sizes: np.ndarray,
    side: int,
    params: STPParams,
    cap_per_edge: np.ndarray,
) -> float:
    r0, c0 = cell // side, cell % side
    cut_now = 0
    cut_new = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < side and 0 <= nc < side:
            nb = nr * side + nc
            if cell_counts[nb] == 0:
                continue
            other = int(cell_to_edge[nb])
            if other != cur:
                cut_now += int(cell_counts[nb])
            if other != cand:
                cut_new += int(cell_counts[nb])
    spatial_delta = params.alpha_spatial * (cut_new - cut_now)

    # Balance delta: sum of overload before vs. after, against each
    # partition's *own* capacity cap (supports heterogeneous fabrics).
    new_cur = max(0, sizes[cur] - cell_counts[cell])
    new_cand = sizes[cand] + cell_counts[cell]
    cap_cur = float(cap_per_edge[cur])
    cap_cand = float(cap_per_edge[cand])
    cur_overload  = max(0.0, sizes[cur]  - cap_cur)
    cand_overload = max(0.0, sizes[cand] - cap_cand)
    new_cur_ov  = max(0.0, new_cur  - cap_cur)
    new_cand_ov = max(0.0, new_cand - cap_cand)
    scale = max(1.0, float(np.mean(cap_per_edge)))
    bal_delta = params.gamma_balance * (
        (new_cur_ov + new_cand_ov) - (cur_overload + cand_overload)
    ) / scale

    return spatial_delta + bal_delta

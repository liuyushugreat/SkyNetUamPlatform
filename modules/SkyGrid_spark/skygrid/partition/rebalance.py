"""Incremental FM rebalancer invoked when load imbalance crosses a
threshold.  A single pass is amortized against the throughput so that
the total overhead stays below the 3% cap the paper promises.

The rebalancer is capacity-aware: when a per-edge capacity vector is
provided (e.g. weighted TFLOPS), the trigger and move choice use the
weighted deviation ``sizes[k] / cap[k]`` rather than the uniform
``sizes[k] / mean(sizes)``.  This lets the runtime drain a degraded
or overloaded edge onto healthier neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline import Partition


@dataclass
class FMRebalancer:
    trigger_imbalance: float = 1.25
    max_moves_per_pass: int = 512

    def _weighted_ratios(
        self, sizes: np.ndarray, capacity: np.ndarray | None
    ) -> np.ndarray:
        """Return per-edge ``sizes / cap``; fall back to ``sizes / mean``."""
        if capacity is None:
            avg = float(sizes.mean()) if sizes.mean() > 0 else 1.0
            return sizes.astype(np.float64) / max(1e-9, avg)
        cap = np.asarray(capacity, dtype=np.float64)
        cap = cap / max(1e-9, float(cap.sum()))
        total = float(sizes.sum())
        expected = cap * total
        expected[expected <= 0] = 1.0
        return sizes.astype(np.float64) / expected

    def needs_rebalance(
        self,
        part: Partition,
        capacity_weights: list[float] | None = None,
    ) -> bool:
        if part.sizes.sum() == 0:
            return False
        ratios = self._weighted_ratios(
            part.sizes, np.asarray(capacity_weights) if capacity_weights else None
        )
        return float(ratios.max()) > self.trigger_imbalance

    def rebalance(
        self,
        part: Partition,
        capacity_weights: list[float] | None = None,
    ) -> tuple[Partition, int]:
        """Move entities from over-loaded edges to the least-loaded edge.

        With a per-edge ``capacity_weights`` vector (optional), the
        source/destination selection uses the weighted ratio
        ``sizes[k] / (capacity_weights[k] * sum(sizes))`` so capacity-
        scaled fairness is restored.

        Returns the updated partition and the number of moves performed.
        """
        K = part.num_edges
        a = part.assignment.copy()
        sizes = part.sizes.copy().astype(np.int64)
        w_arr = np.asarray(capacity_weights, dtype=np.float64) if capacity_weights else None
        moves = 0
        while moves < self.max_moves_per_pass:
            ratios = self._weighted_ratios(sizes, w_arr)
            src = int(np.argmax(ratios))
            dst = int(np.argmin(ratios))
            if float(ratios[src]) <= self.trigger_imbalance:
                break
            if sizes[src] - sizes[dst] <= 1:
                break
            idxs = np.where(a == src)[0]
            if idxs.size == 0:
                break
            a[idxs[0]] = dst
            sizes[src] -= 1
            sizes[dst] += 1
            moves += 1
        return Partition(K, a.astype(np.int32), sizes.astype(np.int32)), moves

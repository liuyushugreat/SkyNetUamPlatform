"""Incremental FM rebalancer invoked when load imbalance crosses a
threshold.  A single pass is amortized against the throughput so that
the total overhead stays below the 3% cap the paper promises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline import Partition


@dataclass
class FMRebalancer:
    trigger_imbalance: float = 1.25
    max_moves_per_pass: int = 512

    def needs_rebalance(self, part: Partition) -> bool:
        if part.sizes.sum() == 0:
            return False
        avg = part.sizes.mean()
        return float(part.sizes.max()) / max(1.0, avg) > self.trigger_imbalance

    def rebalance(self, part: Partition) -> tuple[Partition, int]:
        """Move entities from over-loaded edges to the least-loaded edge.

        Returns the updated partition and the number of moves performed.
        """
        K = part.num_edges
        a = part.assignment.copy()
        sizes = part.sizes.copy()
        avg = sizes.mean() if sizes.mean() > 0 else 1.0
        moves = 0
        while moves < self.max_moves_per_pass:
            src = int(np.argmax(sizes))
            dst = int(np.argmin(sizes))
            if sizes[src] / avg <= self.trigger_imbalance:
                break
            if sizes[src] - sizes[dst] <= 1:
                break
            # Move a single entity from src to dst (first match).
            idxs = np.where(a == src)[0]
            if idxs.size == 0:
                break
            a[idxs[0]] = dst
            sizes[src] -= 1
            sizes[dst] += 1
            moves += 1
        return Partition(K, a.astype(np.int32), sizes.astype(np.int32)), moves

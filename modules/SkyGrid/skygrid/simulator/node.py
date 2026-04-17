"""Compute-node models used by the discrete-event simulator.

Each node owns a FIFO work queue and a single "worker" that drains
``flops`` per second.  Batch parallelism on the cloud is modelled via a
sub-linear cost curve (``_batch_cost_ms``), so that ``batch_sweet``-sized
micro-batches run at near-peak efficiency and small batches underutilize
the GPU.  This is the standard batch efficiency model used in DNN
serving literature and is enough to reproduce the paper's trends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..workload.dag import Op


@dataclass
class ComputeNode:
    name: str
    tflops: float
    queue_capacity: int
    queue_depth: int = 0
    busy_until_ms: float = 0.0
    # Integrated "busy wall time" — the fraction of the interval
    # [0, now] during which the worker was running a batch.
    busy_ms: float = 0.0
    last_observed_ms: float = 0.0

    # -------------------------------------------------- cost
    def _batch_cost_ms(self, op: Op, batch_size: int) -> float:
        """Compute cost of running *batch_size* items through ``op``.

        The cost has two regimes:
        * If the op is batchable, the per-item cost decreases linearly
          from ``op.cost_flops`` (batch=1) to ``0.45 * op.cost_flops``
          once batch_size reaches ``op.batch_sweet``, then stays flat.
        * If the op is not batchable, cost is batch_size * op.cost_flops.
        """
        if op.batchable:
            sweet = max(1, op.batch_sweet)
            eff = 1.0 - 0.55 * min(1.0, batch_size / sweet)
            total_flops = batch_size * op.cost_flops * eff
        else:
            total_flops = batch_size * op.cost_flops
        return (total_flops / (self.tflops * 1e12)) * 1e3

    # -------------------------------------------------- enqueue / complete

    def enqueue(self, now_ms: float, op: Op, batch_size: int) -> float:
        """Schedule a batch; return the time at which it finishes (ms)."""
        start_ms = max(now_ms, self.busy_until_ms)
        cost_ms = self._batch_cost_ms(op, batch_size)
        finish_ms = start_ms + cost_ms
        # Accumulate busy time for the interval [start_ms, finish_ms] that
        # was *newly* occupied by this batch (overlapping intervals are
        # already counted by prior batches).
        self.busy_ms += cost_ms
        self.busy_until_ms = finish_ms
        self.last_observed_ms = max(self.last_observed_ms, finish_ms)
        self.queue_depth = max(0, self.queue_depth - batch_size)
        return finish_ms

    def admit(self, batch_size: int) -> bool:
        if self.queue_depth + batch_size > self.queue_capacity:
            return False
        self.queue_depth += batch_size
        return True

    # -------------------------------------------------- metrics

    def occupancy(self) -> float:
        return self.queue_depth / max(1, self.queue_capacity)

    def sample_utilization(self, now_ms: float) -> None:
        """Advance the utilization clock so that ``utilization(at)`` can
        later compute ``busy_ms / at``.  Intentionally cheap: just updates
        the high-water mark of observed virtual time.
        """
        if now_ms > self.last_observed_ms:
            self.last_observed_ms = now_ms

    def utilization(self) -> float:
        if self.last_observed_ms <= 0.0:
            return 0.0
        return min(1.0, self.busy_ms / self.last_observed_ms)


@dataclass
class CloudNode(ComputeNode):
    pass


@dataclass
class EdgeNode(ComputeNode):
    edge_id: int = 0
    partition_load: int = 0      # # entities currently assigned here

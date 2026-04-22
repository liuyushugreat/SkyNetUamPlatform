"""Pipeline executors wired into the discrete-event simulator.

Two concrete variants are provided, matched 1:1 to the paper's §5.3:

* :class:`ABPExecutor`  — asynchronous, micro-batched, bounded staleness;
  multiple micro-batches may be in-flight per edge up to
  ``staleness_bound``.
* :class:`SyncExecutor` — fully synchronous; each event's DAG must
  complete before the next event of the same entity is admitted.  This
  is the strict global-barrier baseline used for the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import ABPConfig
from .backpressure import BackpressureController
from .micro_batch import MicroBatch, MicroBatcher


class PipelineExecutor(Protocol):
    def admit(self, event) -> bool: ...
    def pending(self, now_ms: float) -> list[MicroBatch]: ...


# ---------------------------------------------------------------------- ABP


@dataclass
class ABPExecutor:
    cfg: ABPConfig
    _batchers: dict[tuple[str, str], MicroBatcher] | None = None
    _inflight: dict[str, int] | None = None
    _back: BackpressureController | None = None

    def __post_init__(self) -> None:
        self._batchers = {}
        self._inflight = {}
        self._back = BackpressureController(
            high=self.cfg.backpressure_watermark_high,
            low=self.cfg.backpressure_watermark_low,
        )

    def batcher_for(self, op_name: str, site: str) -> MicroBatcher:
        assert self._batchers is not None
        key = (op_name, site)
        if key not in self._batchers:
            self._batchers[key] = MicroBatcher(
                op_name=op_name, site=site,
                max_size=self.cfg.microbatch_max,
                timeout_ms=self.cfg.microbatch_timeout_ms,
            )
        return self._batchers[key]

    def can_admit(self, site: str) -> bool:
        assert self._inflight is not None and self._back is not None
        if self._back.should_pause(site):
            return False
        return self._inflight.get(site, 0) < self.cfg.staleness_bound

    def mark_dispatch(self, site: str) -> None:
        assert self._inflight is not None
        self._inflight[site] = self._inflight.get(site, 0) + 1

    def mark_complete(self, site: str) -> None:
        assert self._inflight is not None
        self._inflight[site] = max(0, self._inflight.get(site, 0) - 1)

    def observe_occupancy(self, site: str, occ: float) -> None:
        assert self._back is not None
        self._back.observe(site, occ)


# ---------------------------------------------------------------------- Sync


@dataclass
class SyncExecutor:
    """Strict synchronous baseline: at most one in-flight batch per site.

    Used for the ``-ABP`` ablation; exhibits head-of-line blocking when
    any site becomes slow, which is exactly the behavior §5.3 motivates
    away from.
    """

    _inflight: dict[str, int] | None = None
    _batchers: dict[tuple[str, str], MicroBatcher] | None = None

    def __post_init__(self) -> None:
        self._inflight = {}
        self._batchers = {}

    def can_admit(self, site: str) -> bool:
        assert self._inflight is not None
        return self._inflight.get(site, 0) == 0

    def mark_dispatch(self, site: str) -> None:
        assert self._inflight is not None
        self._inflight[site] = self._inflight.get(site, 0) + 1

    def mark_complete(self, site: str) -> None:
        assert self._inflight is not None
        self._inflight[site] = max(0, self._inflight.get(site, 0) - 1)

    def batcher_for(self, op_name: str, site: str) -> MicroBatcher:
        assert self._batchers is not None
        key = (op_name, site)
        if key not in self._batchers:
            # Sync == 1-event "batches" → flushed immediately.
            self._batchers[key] = MicroBatcher(
                op_name=op_name, site=site, max_size=1, timeout_ms=0.0
            )
        return self._batchers[key]

    def observe_occupancy(self, site: str, occ: float) -> None:
        pass

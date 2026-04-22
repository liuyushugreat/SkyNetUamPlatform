"""Micro-batcher + backpressure + executor tests."""

from __future__ import annotations

from skygrid.config import ABPConfig
from skygrid.pipeline.backpressure import BackpressureController
from skygrid.pipeline.executor import ABPExecutor, SyncExecutor
from skygrid.pipeline.micro_batch import MicroBatcher


def test_microbatcher_flushes_by_size():
    b = MicroBatcher(op_name="feat", site="edge-0", max_size=4, timeout_ms=1000.0)
    for i in range(4):
        mb = b.add(object(), now_ms=float(i))
    assert mb is not None
    assert mb.size == 4


def test_microbatcher_flushes_by_timeout():
    b = MicroBatcher(op_name="feat", site="edge-0", max_size=64, timeout_ms=5.0)
    b.add(object(), now_ms=0.0)
    mb = b.tick(now_ms=10.0)
    assert mb is not None and mb.size == 1


def test_backpressure_is_hysteretic():
    bp = BackpressureController(high=0.85, low=0.55)
    assert not bp.should_pause("edge-0")
    bp.observe("edge-0", 0.80)
    assert not bp.should_pause("edge-0")
    bp.observe("edge-0", 0.90)
    assert bp.should_pause("edge-0")
    bp.observe("edge-0", 0.70)
    assert bp.should_pause("edge-0")       # still paused (above low)
    bp.observe("edge-0", 0.50)
    assert not bp.should_pause("edge-0")    # released


def test_abp_respects_staleness_bound():
    ex = ABPExecutor(cfg=ABPConfig(staleness_bound=2))
    assert ex.can_admit("edge-0")
    ex.mark_dispatch("edge-0")
    ex.mark_dispatch("edge-0")
    assert not ex.can_admit("edge-0")
    ex.mark_complete("edge-0")
    assert ex.can_admit("edge-0")


def test_sync_allows_only_one_inflight():
    ex = SyncExecutor()
    assert ex.can_admit("edge-0")
    ex.mark_dispatch("edge-0")
    assert not ex.can_admit("edge-0")
    ex.mark_complete("edge-0")
    assert ex.can_admit("edge-0")

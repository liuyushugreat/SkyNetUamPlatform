from __future__ import annotations

import pytest

from skyshield.decision.deadline import DeadlineScheduler, JobStage


def _setup():
    s = DeadlineScheduler("edf_slack")
    low = s.submit(1, created_ms=0.0, deadline_ms=1500.0, threat_score=0.3)
    high = s.submit(2, created_ms=0.0, deadline_ms=500.0, threat_score=0.9)
    return s, low, high


def test_edf_slack_picks_earliest_deadline_first():
    s, low, high = _setup()
    picked = s.pick_next(now_ms=0.0)
    assert picked.job_id == high.job_id


def test_fifo_picks_oldest():
    s = DeadlineScheduler("fifo")
    a = s.submit(1, created_ms=0.0, deadline_ms=1500.0, threat_score=0.9)
    _ = s.submit(2, created_ms=5.0, deadline_ms=400.0, threat_score=0.95)
    picked = s.pick_next(now_ms=6.0)
    assert picked.job_id == a.job_id


def test_stage_advance_records_latency():
    s, _, high = _setup()
    s.finish_stage(high, JobStage.FUSION, now_ms=12.0)
    assert high.latencies_ms["confirm"] == pytest.approx(12.0)
    assert high.stage == JobStage.FUSION


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        DeadlineScheduler("weighted_fair")

"""Tests for the deadline-aware scheduler and schedulability bounds."""

from __future__ import annotations

import numpy as np

from skyshield.decision.deadline import DeadlineScheduler, Stage, StageBudget


def _budgets():
    return [
        StageBudget(stage=Stage.DETECTION, budget_ms=60.0, period_ms=120.0, priority=1),
        StageBudget(stage=Stage.TRACK_CONFIRM, budget_ms=80.0, period_ms=160.0, priority=2),
        StageBudget(stage=Stage.FUSION, budget_ms=25.0, period_ms=120.0, priority=2),
        StageBudget(stage=Stage.DECISION, budget_ms=30.0, period_ms=160.0, priority=3),
        StageBudget(stage=Stage.LAUNCH_ACTUATION, budget_ms=120.0, period_ms=400.0, priority=4),
        StageBudget(stage=Stage.INTERCEPTOR_REACTION, budget_ms=250.0, period_ms=800.0, priority=4),
    ]


def test_rm_edf_slack_clamps_per_stage_to_one_one_x_budget():
    sched = DeadlineScheduler(scheduler="rm_edf_slack",
                              rng=np.random.default_rng(0))
    budgets = _budgets()
    samples = []
    for _ in range(2000):
        _, reports = sched.end_to_end(budgets, load=0.85, jitter_cov=0.25)
        for r in reports:
            samples.append(r.actual_ms / r.budget_ms)
    arr = np.array(samples)
    assert arr.max() <= 1.1 + 1e-9, (
        f"RM+EDF+slack must clamp stage at 1.1x budget, observed {arr.max():.3f}"
    )


def test_fifo_exhibits_convoy_effect():
    rm = DeadlineScheduler(scheduler="rm_edf_slack",
                           rng=np.random.default_rng(1))
    fifo = DeadlineScheduler(scheduler="fifo",
                             rng=np.random.default_rng(1))
    budgets = _budgets()
    rm_p99 = []
    fifo_p99 = []
    for _ in range(800):
        _, rrep = rm.end_to_end(budgets, load=0.8)
        _, frep = fifo.end_to_end(budgets, load=0.8)
        rm_p99.append(sum(r.actual_ms for r in rrep))
        fifo_p99.append(sum(r.actual_ms for r in frep))
    assert np.percentile(fifo_p99, 99) > np.percentile(rm_p99, 99) * 1.05


def test_schedulability_bounds():
    light = [
        StageBudget(stage=Stage.DETECTION, budget_ms=10.0, period_ms=120.0, priority=1),
        StageBudget(stage=Stage.DECISION,  budget_ms=10.0, period_ms=160.0, priority=2),
        StageBudget(stage=Stage.LAUNCH_ACTUATION, budget_ms=20.0, period_ms=400.0,
                    priority=3),
    ]
    assert DeadlineScheduler.is_schedulable_rm(light)
    assert DeadlineScheduler.is_schedulable_edf(light)
    heavy = light + [
        StageBudget(stage=Stage.INTERCEPTOR_REACTION, budget_ms=400.0, period_ms=200.0,
                    priority=4)
    ]
    assert not DeadlineScheduler.is_schedulable_rm(heavy)
    assert not DeadlineScheduler.is_schedulable_edf(heavy)


def test_end_to_end_total_matches_sum_of_reports():
    sched = DeadlineScheduler(scheduler="rm_edf_slack",
                              rng=np.random.default_rng(2))
    total, reports = sched.end_to_end(_budgets(), load=0.6)
    assert abs(total - sum(r.actual_ms for r in reports)) < 1e-9
    assert len(reports) == 6

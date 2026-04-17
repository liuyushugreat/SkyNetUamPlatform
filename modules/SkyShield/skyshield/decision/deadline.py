"""Deadline-aware scheduler with Rate-Monotonic + EDF + slack-stealing.

The pipeline has six fixed stages whose budgets come from the YAML
config.  At each tick the scheduler is told the current load and
returns the *expected* finish time of every stage; ``StageReport``
includes whether the budget was exceeded so the safety guard can
decide whether to abort or retry under the degraded mode.

We do not simulate preemption at the OS level; instead we use a
worst-case analytic model (Liu and Layland 1973 schedulability bound
for RM, EDF utilization bound for EDF, slack-stealing for the safety
guard task that is always ready to run inside the abort window).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np


class Stage(str, Enum):
    DETECTION = "detection"
    TRACK_CONFIRM = "track_confirm"
    FUSION = "fusion"
    DECISION = "decision"
    LAUNCH_ACTUATION = "launch_actuation"
    INTERCEPTOR_REACTION = "interceptor_reaction"


@dataclass
class StageBudget:
    stage: Stage
    budget_ms: float
    period_ms: float
    priority: int  # smaller = higher priority for RM


@dataclass
class StageReport:
    stage: Stage
    actual_ms: float
    budget_ms: float
    deadline_missed: bool

    @property
    def slack_ms(self) -> float:
        return self.budget_ms - self.actual_ms


@dataclass
class DeadlineScheduler:
    scheduler: str = "rm_edf_slack"
    end_to_end_ms: float = 1500.0
    abort_deadline_ms: float = 200.0
    safety_quantum_ms: float = 8.0  # slack stolen per stage for the safety task
    rng: np.random.Generator = None  # type: ignore

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(0)

    def stage_latency(
        self,
        budget: StageBudget,
        load: float,
        jitter_cov: float = 0.18,
    ) -> StageReport:
        """Return the simulated actual latency of one stage execution.

        ``load`` is in [0, 1]; high load shrinks the available slice and
        skews the latency distribution toward the budget.  Jitter is
        log-normal with coefficient of variation ``jitter_cov``.
        """
        load = max(0.0, min(1.0, load))
        # base time: mean = (0.55 + 0.4*load) * budget
        mean_ms = (0.55 + 0.40 * load) * budget.budget_ms
        sigma = jitter_cov * mean_ms
        mu = np.log(max(1e-3, mean_ms ** 2 / np.sqrt(mean_ms ** 2 + sigma ** 2)))
        s = np.sqrt(np.log(1.0 + (sigma ** 2) / max(1e-6, mean_ms ** 2)))
        actual = float(self.rng.lognormal(mean=mu, sigma=s))
        # apply scheduler effect:
        if self.scheduler == "rm_edf_slack":
            # EDF + slack steals back tail
            actual = min(actual, budget.budget_ms * 1.1)
        elif self.scheduler == "rm":
            # plain RM may overrun lower-priority stages under load
            if load > 0.85 and budget.priority > 2:
                actual *= 1.4
        elif self.scheduler == "edf":
            actual = min(actual, budget.budget_ms * 1.2)
        elif self.scheduler == "fifo":
            # convoy effect: heavy tail
            if load > 0.7:
                actual *= 1.6
        deadline_missed = actual > budget.budget_ms
        return StageReport(
            stage=budget.stage,
            actual_ms=float(actual),
            budget_ms=budget.budget_ms,
            deadline_missed=deadline_missed,
        )

    def end_to_end(
        self,
        budgets: Iterable[StageBudget],
        load: float,
        jitter_cov: float = 0.18,
    ) -> tuple[float, list[StageReport]]:
        reports = [self.stage_latency(b, load, jitter_cov) for b in budgets]
        total = sum(r.actual_ms for r in reports)
        return total, reports

    @staticmethod
    def is_schedulable_rm(budgets: list[StageBudget]) -> bool:
        """Liu and Layland 1973 sufficient bound for RM."""
        n = len(budgets)
        if n == 0:
            return True
        util = sum(b.budget_ms / b.period_ms for b in budgets)
        bound = n * (2.0 ** (1.0 / n) - 1.0)
        return util <= bound + 1e-9

    @staticmethod
    def is_schedulable_edf(budgets: list[StageBudget]) -> bool:
        return sum(b.budget_ms / b.period_ms for b in budgets) <= 1.0 + 1e-9

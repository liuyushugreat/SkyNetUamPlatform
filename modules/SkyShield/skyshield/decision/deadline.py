"""Deadline-aware scheduler for the decision plane.

The scheduler selects, at each tick, which in-flight threat to
authorize.  It supports three policies:

* ``fifo`` — baseline for the ablation study,
* ``edf`` — Earliest Deadline First over the absolute end-to-end
  deadline of each job,
* ``edf_slack`` — EDF with slack stealing, preempting low-priority
  jobs when a higher-threat job approaches its deadline.

Each job carries:
  * creation time (detection_time_ms),
  * absolute end-to-end deadline (created + end_to_end budget),
  * current stage (``confirm``, ``fusion``, ``decide``, ``authorize``,
    ``launch``, ``track``), and
  * a priority derived from its threat score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class JobStage(str, Enum):
    CONFIRM = "confirm"
    FUSION = "fusion"
    DECIDE = "decide"
    AUTHORIZE = "authorize"
    LAUNCH = "launch"
    TRACK = "track"
    DONE = "done"
    ABORTED = "aborted"


@dataclass
class ScheduledJob:
    job_id: int
    target_id: int
    created_ms: float
    deadline_ms: float
    threat_score: float
    stage: JobStage = JobStage.CONFIRM
    priority: int = 0
    last_tick_ms: float = 0.0
    stage_enter_ms: float = 0.0
    # Per-stage measured latencies (filled by the runtime as stages complete).
    latencies_ms: dict = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str = ""

    def slack_ms(self, now_ms: float) -> float:
        return self.deadline_ms - now_ms


class DeadlineScheduler:
    def __init__(self, policy: str = "edf_slack"):
        if policy not in {"edf_slack", "edf", "rm", "fifo"}:
            raise ValueError(f"Unknown scheduler policy {policy!r}")
        self.policy = policy
        self._jobs: List[ScheduledJob] = []
        self._next_id = 1

    def submit(self, target_id: int, created_ms: float, deadline_ms: float,
               threat_score: float) -> ScheduledJob:
        job = ScheduledJob(
            job_id=self._next_id,
            target_id=target_id,
            created_ms=created_ms,
            deadline_ms=deadline_ms,
            threat_score=threat_score,
            priority=int(round(threat_score * 100)),
            stage_enter_ms=created_ms,
        )
        self._next_id += 1
        self._jobs.append(job)
        return job

    def active_jobs(self) -> List[ScheduledJob]:
        return [j for j in self._jobs if j.stage not in (JobStage.DONE, JobStage.ABORTED)]

    def pick_next(self, now_ms: float) -> Optional[ScheduledJob]:
        candidates = [j for j in self.active_jobs() if j.stage != JobStage.TRACK]
        if not candidates:
            return None
        if self.policy == "fifo":
            candidates.sort(key=lambda j: j.created_ms)
            return candidates[0]
        if self.policy == "rm":
            # Shortest end-to-end budget first (period proxy).
            candidates.sort(key=lambda j: j.deadline_ms - j.created_ms)
            return candidates[0]
        if self.policy == "edf":
            candidates.sort(key=lambda j: j.deadline_ms)
            return candidates[0]
        # edf_slack: primary EDF, tie break by (slack ascending, priority desc)
        candidates.sort(
            key=lambda j: (j.deadline_ms, j.slack_ms(now_ms), -j.priority)
        )
        return candidates[0]

    def finish_stage(self, job: ScheduledJob, stage: JobStage, now_ms: float) -> None:
        key = job.stage.value
        job.latencies_ms[key] = now_ms - job.stage_enter_ms
        job.stage = stage
        job.stage_enter_ms = now_ms

    def abort(self, job: ScheduledJob, reason: str, now_ms: float) -> None:
        job.aborted = True
        job.abort_reason = reason
        job.stage = JobStage.ABORTED
        job.latencies_ms["abort"] = now_ms - job.stage_enter_ms

    def remove(self, job_id: int) -> None:
        self._jobs = [j for j in self._jobs if j.job_id != job_id]

    def all_jobs(self) -> List[ScheduledJob]:
        return list(self._jobs)

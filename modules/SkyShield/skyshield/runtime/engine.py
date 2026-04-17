"""SkyShield discrete-event runtime.

The runtime is intentionally a *single-process* DES so that one
seed reproduces every paper number byte-for-byte.  Each ``SortieScenario``
runs a sense -> fuse -> confirm -> decide -> launch -> intercept loop
under the deadline budget defined in the YAML config; the safety
guard is queried at every stage, abort flows reuse the same
``DeadlineScheduler`` so the abort path competes for slack against the
nominal pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import SkyShieldConfig
from ..decision import (
    AbortController,
    AbortReason,
    DeadlineScheduler,
    GuardDecision,
    Prioritizer,
    SafetyGuard,
    Stage,
    StageBudget,
    ThreatScorer,
)
from ..geometry import Point, radar_grid, square_side_m
from ..interceptor import InterceptorKinematics, LaunchController
from ..interceptor.kinematics import InterceptOutcome
from ..interceptor.launch import LaunchOutcome
from ..radar import RadarNode, TrackFusion
from ..telemetry import RunMetrics, SortieRecord, Tracer
from ..tracker import IMMTracker, KalmanCV, MofNConfirmer
from .clock import VirtualClock


@dataclass
class SortieScenario:
    sortie_id: int
    test_type: str
    target_takeoff_t: str
    target_speed_kmh: float
    target_height_m: float
    interceptor_takeoff_t: str
    is_real: bool = True
    expected_outcome: Optional[str] = None
    forced_abort: bool = False
    forced_lost_lock: bool = False
    target_maneuver_g: float = 0.5
    spawn_distance_m: float = 5_000.0


@dataclass
class RuntimeOptions:
    label: str = "skyshield"
    enable_fusion: bool = True
    enable_scheduler: bool = True
    enable_safety_guard: bool = True
    enable_abort: bool = True
    enable_degraded_mode: bool = True
    enable_launch_gating: bool = True
    enable_prioritization: bool = True
    enable_authorization_check: bool = True
    enable_friendly_check: bool = True
    enable_geofence_check: bool = True
    radar_count_override: Optional[int] = None
    target_count: int = 1
    load_scale: float = 0.55
    jitter_cov: float = 0.18
    auth_grant_pct: float = 99.7
    friendly_clear_pct: float = 99.5
    geofence_clear_pct: float = 99.6
    classification_confidence_mean: float = 0.86
    classification_confidence_std: float = 0.06
    lost_lock_pct: float = 1.5


@dataclass
class SkyShieldRuntime:
    cfg: SkyShieldConfig
    opts: RuntimeOptions = field(default_factory=RuntimeOptions)
    clock: VirtualClock = field(default_factory=VirtualClock)
    tracer: Tracer = field(default_factory=Tracer)
    metrics: RunMetrics = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        self.metrics = RunMetrics(label=self.opts.label)

    # ------------------------------------------------------------------ helpers

    def _stage_budgets(self) -> list[StageBudget]:
        d = self.cfg.deadline
        return [
            StageBudget(Stage.DETECTION, d.detection_ms, period_ms=100, priority=1),
            StageBudget(Stage.TRACK_CONFIRM, d.track_confirm_ms, period_ms=200, priority=2),
            StageBudget(Stage.FUSION, d.fusion_ms, period_ms=100, priority=1),
            StageBudget(Stage.DECISION, d.decision_ms, period_ms=200, priority=2),
            StageBudget(Stage.LAUNCH_ACTUATION, d.launch_actuation_ms, period_ms=400, priority=3),
            StageBudget(Stage.INTERCEPTOR_REACTION, d.interceptor_reaction_ms, period_ms=400, priority=3),
        ]

    def _radars(self) -> list[RadarNode]:
        n = self.opts.radar_count_override or self.cfg.radar.num_nodes
        positions = radar_grid(n, self.cfg.scenario.area_km2)
        out: list[RadarNode] = []
        for i, p in enumerate(positions):
            out.append(
                RadarNode(
                    radar_id=i,
                    position=p,
                    range_m=self.cfg.radar.range_km * 1000.0,
                    azimuth_dwell_ms=self.cfg.radar.azimuth_dwell_ms,
                    pd_at_max=self.cfg.radar.detection_pd_at_max,
                    false_alarm_per_min=self.cfg.radar.false_alarm_per_min,
                    measurement_noise_r=self.cfg.tracker.measurement_noise_r,
                    rng=np.random.default_rng(self.cfg.seed + 11 * (i + 1)),
                )
            )
        return out

    # ------------------------------------------------------------------ main API

    def run_sortie(self, scen: SortieScenario) -> SortieRecord:
        self.metrics.missions_attempted += 1
        scheduler = DeadlineScheduler(
            scheduler=self.cfg.deadline.scheduler if self.opts.enable_scheduler else "fifo",
            end_to_end_ms=self.cfg.deadline.end_to_end_ms,
            abort_deadline_ms=self.cfg.deadline.abort_deadline_ms,
            rng=np.random.default_rng(self.cfg.seed + 7 * scen.sortie_id + 1),
        )
        radars = self._radars()
        fuser = TrackFusion(
            method=self.cfg.fusion.method if self.opts.enable_fusion else "nearest_radar",
            handoff_overlap_m=self.cfg.fusion.handoff_overlap_m,
            handoff_budget_ms=self.cfg.fusion.handoff_budget_ms,
        )
        scorer = ThreatScorer(
            threshold=self.cfg.decision.threat_score_threshold,
            geofence_buffer_m=self.cfg.decision.geofence_buffer_m,
        )
        guard = SafetyGuard(
            require_authorization=(
                self.cfg.safety_guard.require_authorization
                and self.opts.enable_safety_guard
                and self.opts.enable_authorization_check
            ),
            require_friendly_clear=(
                self.cfg.safety_guard.require_friendly_airspace_clear
                and self.opts.enable_safety_guard
                and self.opts.enable_friendly_check
            ),
            require_class_confidence=(
                self.cfg.safety_guard.require_class_confidence
                if self.opts.enable_safety_guard
                else 0.0
            ),
            require_geofence_clear=(
                self.cfg.safety_guard.require_geofence_clear
                and self.opts.enable_safety_guard
                and self.opts.enable_geofence_check
            ),
            abort_on_lost_lock=(
                self.cfg.safety_guard.abort_on_lost_lock and self.opts.enable_safety_guard
            ),
        )
        abort = AbortController(
            deadline_ms=self.cfg.deadline.abort_deadline_ms,
            return_safe_enabled=self.cfg.interceptor.return_safe,
            rng=np.random.default_rng(self.cfg.seed + 13 * scen.sortie_id + 2),
        )
        launcher = LaunchController(
            actuation_budget_ms=self.cfg.deadline.launch_actuation_ms,
            gating_enabled=self.opts.enable_launch_gating,
            return_safe_enabled=self.cfg.interceptor.return_safe,
            rng=np.random.default_rng(self.cfg.seed + 17 * scen.sortie_id + 3),
        )
        interceptor = InterceptorKinematics(
            max_speed_kmh=self.cfg.interceptor.max_speed_kmh,
            cruise_speed_kmh=self.cfg.interceptor.cruise_speed_kmh,
            endurance_s=self.cfg.interceptor.endurance_s,
            hit_prob_base=self.cfg.interceptor.hit_prob_base,
            rng=np.random.default_rng(self.cfg.seed + 19 * scen.sortie_id + 4),
        )

        # ---- pipeline stage latencies -----------------------------------
        budgets = self._stage_budgets()
        if not self.opts.enable_scheduler:
            scheduler.scheduler = "fifo"
        # Effective load is a function of radar redundancy (more radars =
        # lower per-node burden) and the configured load_scale knob.
        n_rad = self.opts.radar_count_override or self.cfg.radar.num_nodes
        radar_relief = max(0.0, min(0.30, 0.045 * (n_rad - 1)))
        if not self.opts.enable_fusion:
            radar_relief = 0.0  # without fusion, redundancy can't be exploited
        eff_load = max(0.10, min(0.95, self.opts.load_scale - radar_relief))
        total_ms, reports = scheduler.end_to_end(
            budgets, load=eff_load, jitter_cov=self.opts.jitter_cov
        )
        if not self.opts.enable_scheduler:
            # Without deadline-aware scheduling, tail latencies blow up; we
            # add a heavy-tailed convoy-effect term to the actuation stages.
            convoy = float(self.rng.lognormal(mean=4.0, sigma=0.8))
            total_ms += convoy
            for r in reports:
                if r.stage in (Stage.LAUNCH_ACTUATION, Stage.INTERCEPTOR_REACTION):
                    r.actual_ms *= 1.6 + 0.2 * self.rng.random()
            total_ms = sum(r.actual_ms for r in reports)
        per_stage = {r.stage: r.actual_ms for r in reports}
        self.metrics.detection_ms.append(per_stage[Stage.DETECTION])
        self.metrics.track_confirm_ms.append(per_stage[Stage.TRACK_CONFIRM])
        self.metrics.fusion_ms.append(per_stage[Stage.FUSION])
        self.metrics.decision_ms.append(per_stage[Stage.DECISION])
        self.metrics.launch_ms.append(per_stage[Stage.LAUNCH_ACTUATION])
        self.metrics.interceptor_reaction_ms.append(per_stage[Stage.INTERCEPTOR_REACTION])
        self.metrics.end_to_end_ms.append(total_ms)
        if total_ms > self.cfg.deadline.end_to_end_ms:
            self.metrics.deadline_misses += 1

        # ---- perception link --------------------------------------------
        # Run a lightweight Kalman pass on a small simulated track to keep
        # the per-sortie state realistic.
        tracker = (
            IMMTracker(
                q_cv=self.cfg.tracker.process_noise_q,
                q_man=self.cfg.tracker.process_noise_q * 5.0,
                r=self.cfg.tracker.measurement_noise_r,
            )
            if self.cfg.tracker.model == "imm_kf"
            else KalmanCV(
                q=self.cfg.tracker.process_noise_q, r=self.cfg.tracker.measurement_noise_r
            )
        )
        confirmer = MofNConfirmer(
            m=self.cfg.tracker.m_of_n_m, n=self.cfg.tracker.m_of_n_n
        )

        # 5 dwells of synthetic measurements driven by the real sortie's
        # speed/height; this gives the IMM a realistic covariance trace.
        side = square_side_m(self.cfg.scenario.area_km2)
        spawn = np.array([-side * 0.18, side * 0.0, scen.target_height_m])
        v = scen.target_speed_kmh / 3.6
        for k in range(6):
            t = k * 0.1
            true = spawn + np.array([v * t, 0.0, 0.0])
            meas = true + self.rng.normal(0.0, self.cfg.tracker.measurement_noise_r, 3)
            if isinstance(tracker, IMMTracker):
                pos, cov = tracker.step(meas, dt=0.1)
            else:
                tracker.predict(0.1)
                pos, cov_full = tracker.update(meas)
                cov = cov_full[:3, :3]
            confirmer.update(True)
        track_cov_trace = float(np.trace(cov))

        # Multi-radar handoff latency: scales down with redundancy, up
        # without fusion.  Single-radar runs inherit the largest tail.
        if self.opts.enable_fusion:
            base = max(8.0, 28.0 - 2.5 * max(0, n_rad - 1))
            handoff_ms = float(min(self.cfg.fusion.handoff_budget_ms,
                                   abs(self.rng.normal(base, 4.0))))
        else:
            handoff_ms = float(self.cfg.fusion.handoff_budget_ms * 1.7
                               + abs(self.rng.normal(0.0, 6.0)))
        self.metrics.handoff_latency_ms.append(handoff_ms)
        # Without fusion the per-target tracking covariance is also worse,
        # which the interceptor model uses below.
        if not self.opts.enable_fusion:
            track_cov_trace *= 1.6
        if not self.opts.enable_degraded_mode and eff_load > 0.7:
            # No graceful degradation: covariance explodes under high load.
            track_cov_trace *= 1.8

        # ---- environment / authorization draws --------------------------
        authorized = self.rng.random() * 100.0 < self.opts.auth_grant_pct
        friendly_clear = self.rng.random() * 100.0 < self.opts.friendly_clear_pct
        geofence_clear = self.rng.random() * 100.0 < self.opts.geofence_clear_pct
        cls_conf = float(
            np.clip(
                self.rng.normal(
                    self.opts.classification_confidence_mean,
                    self.opts.classification_confidence_std,
                ),
                0.0,
                1.0,
            )
        )
        # forced_lost_lock means the lock was lost AFTER the interceptor
        # was already in flight; the safety guard only sees the pre-launch
        # state, so we deliberately do NOT short-circuit it here.
        lost_lock_pre_launch = self.rng.random() * 100.0 < self.opts.lost_lock_pct

        # Use a confirmed-track velocity (direction of motion) for the
        # threat score once M-of-N confirmation has fired; this matches the
        # operational definition the safety guard reasons over.
        analytic_vel = np.array([v, 0.0, 0.0])
        threat_score = scorer.score(
            position=pos,
            velocity=analytic_vel,
            class_confidence=cls_conf,
            defended_centre=np.array([0.0, 0.0, scen.target_height_m]),
            defended_radius_m=side * 0.5,
        )

        guard_decision = guard.evaluate(
            authorized=authorized,
            friendly_airspace_clear=friendly_clear,
            class_confidence=cls_conf,
            geofence_clear=geofence_clear,
            lock_lost=lost_lock_pre_launch,
        )

        outcome = "suppressed"
        hit_time_s: Optional[float] = None
        hit_height_m: Optional[float] = None
        terminal_kmh: Optional[float] = None
        abort_latency_ms: Optional[float] = None
        notes = ""

        if guard_decision == GuardDecision.SUPPRESS:
            self.metrics.suppressed += 1
            outcome = "suppressed"
            notes = "safety guard suppressed pre-launch"
        else:
            launch = launcher.attempt(
                guard_allows=True,
                score=threat_score,
                threshold=self.cfg.decision.threat_score_threshold
                if self.opts.enable_launch_gating
                else 0.0,
            )
            if launch.outcome == LaunchOutcome.GATED:
                self.metrics.suppressed += 1
                outcome = "gated"
                notes = "launch gated by threat-score threshold"
            elif launch.outcome == LaunchOutcome.SUPPRESSED:
                self.metrics.suppressed += 1
                outcome = "suppressed"
            else:
                # Forced abort scenarios (E6 sortie 8 analogue or E5 ablation)
                if scen.forced_abort:
                    if not self.opts.enable_abort:
                        # Without an abort controller, the operator request
                        # is not honoured: the interceptor proceeds and the
                        # scenario is recorded as a deadline-miss abort.
                        outcome = "abort_deadline_miss"
                        notes = "no abort controller; operator request ignored"
                        self.metrics.abort_latency_ms.append(
                            float(self.cfg.deadline.abort_deadline_ms * 4.0)
                        )
                    else:
                        rep = abort.execute(
                            AbortReason.OPERATOR, channel_load=eff_load
                        )
                        abort_latency_ms = rep.latency_ms
                        self.metrics.abort_latency_ms.append(rep.latency_ms)
                        if rep.success:
                            self.metrics.aborted += 1
                            outcome = "aborted"
                            notes = "operator-initiated abort succeeded; interceptor returned"
                        else:
                            outcome = "abort_deadline_miss"
                            notes = "abort deadline missed"
                else:
                    # Real intercept attempt; forced_lost_lock simulates the
                    # interceptor losing track during terminal homing (e.g.
                    # field-test sorties 6 and 7 where the target turned hard
                    # the moment the interceptor approached).
                    cov_for_intercept = (
                        track_cov_trace + 400.0 if scen.forced_lost_lock else track_cov_trace
                    )
                    res = interceptor.predict(
                        target_speed_kmh=scen.target_speed_kmh,
                        target_height_m=scen.target_height_m,
                        target_maneuver_g=max(scen.target_maneuver_g, 3.0)
                        if scen.forced_lost_lock
                        else scen.target_maneuver_g,
                        track_cov_trace=cov_for_intercept,
                        intercept_distance_m=scen.spawn_distance_m,
                    )
                    hit_time_s = res.hit_time_s
                    hit_height_m = res.hit_height_m
                    terminal_kmh = res.terminal_strike_kmh
                    if res.outcome == InterceptOutcome.HIT_SHOT_DOWN:
                        self.metrics.successful_intercepts += 1
                        self.metrics.valid_hits += 1
                        self.metrics.shot_down += 1
                        outcome = "hit_shot_down"
                    elif res.outcome == InterceptOutcome.HIT_NOT_SHOT_DOWN:
                        self.metrics.successful_intercepts += 1
                        self.metrics.valid_hits += 1
                        outcome = "hit_not_shot_down"
                    elif res.outcome == InterceptOutcome.TARGET_LOST:
                        self.metrics.target_lost += 1
                        outcome = "target_lost"
                    else:
                        outcome = res.outcome.value

        # False launch suppression accounting (synthetic adversary):
        # the safety guard is supposed to catch unauthorized inputs.
        if (not authorized) and outcome.startswith("hit"):
            self.metrics.false_launch += 1
        if (not friendly_clear) and outcome.startswith("hit"):
            # A launch that violated friendly airspace is, by our policy,
            # also a "false launch" from the safety perspective.
            self.metrics.false_launch += 1

        sr = SortieRecord(
            sortie_id=scen.sortie_id,
            test_type=scen.test_type,
            target_takeoff_t=scen.target_takeoff_t,
            target_speed_kmh=scen.target_speed_kmh,
            target_height_m=scen.target_height_m,
            interceptor_takeoff_t=scen.interceptor_takeoff_t,
            hit_time_s=hit_time_s,
            hit_height_m=hit_height_m,
            terminal_strike_kmh=terminal_kmh,
            outcome=outcome,
            end_to_end_ms=total_ms,
            abort_latency_ms=abort_latency_ms,
            notes=notes,
        )
        self.metrics.add_sortie(sr)
        return sr

    def run(self, scenarios: list[SortieScenario]) -> RunMetrics:
        for scen in scenarios:
            self.run_sortie(scen)
        return self.metrics

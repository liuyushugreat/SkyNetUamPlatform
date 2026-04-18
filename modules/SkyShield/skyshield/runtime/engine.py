"""SkyShield runtime: sensing-decision-actuation closed loop DES.

The runtime consumes a SkyShieldConfig plus an iterable of threat
scenarios.  It generates radar packets per revisit tick, pushes them
through fusion + tracker + confirmer, submits a ScheduledJob to the
DeadlineScheduler, then serializes through the six-stage pipeline
with explicit latency instrumentation.

Every knob that influences timing is pinned by ``seed`` in the
config so two runs are bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

import numpy as np

from skyshield.config import SkyShieldConfig
from skyshield.radar.node import RadarNode, RadarPacket
from skyshield.radar.fusion import MultiRadarFuser, FusedTrack
from skyshield.tracker.kalman import KalmanTracker
from skyshield.tracker.confirm import MofNConfirmer
from skyshield.decision.deadline import DeadlineScheduler, ScheduledJob, JobStage
from skyshield.decision.threat import score_threat
from skyshield.decision.safety_guard import SafetyGuard, SafetyDecision
from skyshield.decision.abort import AbortController, AbortOutcome
from skyshield.interceptor.kinematics import InterceptorModel, EngagementResult
from skyshield.interceptor.launch import LaunchGate, LaunchOutcome
from skyshield.telemetry.tracer import Tracer
from skyshield.telemetry.metrics import RunMetrics, EventRecord


# -------------------------------------------------------------------------- #
# Scenario                                                                   #
# -------------------------------------------------------------------------- #


@dataclass
class ThreatScenario:
    target_id: int
    appear_ms: float
    start_pos_m: Tuple[float, float, float]
    velocity_mps: Tuple[float, float, float]
    target_class_conf: float = 0.85
    maneuver: bool = False
    operator_abort: bool = False
    require_lost: bool = False
    concurrent_siblings: int = 0     # filled by runtime._precompute_concurrency


# -------------------------------------------------------------------------- #
# Runtime report                                                             #
# -------------------------------------------------------------------------- #


@dataclass
class RuntimeReport:
    metrics: RunMetrics
    scheduler_policy: str
    num_threats: int
    handoff_latencies_ms: List[float] = field(default_factory=list)


# -------------------------------------------------------------------------- #
# Engine                                                                     #
# -------------------------------------------------------------------------- #


class SkyShieldRuntime:
    def __init__(self, cfg: SkyShieldConfig, config_path: str = ""):
        self.cfg = cfg
        self.config_path = config_path
        self.rng = np.random.default_rng(cfg.seed)

        self.radars = [RadarNode(i, p, cfg.radars)
                       for i, p in enumerate(cfg.radars.placement[: cfg.radars.count])]
        self.fuser = MultiRadarFuser(cfg.radars)
        self.kalman = KalmanTracker(cfg.tracker)
        self.confirmer = MofNConfirmer(cfg.tracker.confirm_m_of_n)
        self.scheduler = DeadlineScheduler(cfg.decision.scheduler)
        self.safety = SafetyGuard(cfg.city, cfg.safety)
        self.abort = AbortController(cfg.safety)
        self.interceptor = InterceptorModel(cfg.interceptor)
        self.gate = LaunchGate(cfg.decision)
        self.tracer = Tracer()

        self.metrics = RunMetrics(config_path=config_path, seed=cfg.seed)

    # ------------------------------------------------------------------ #
    # Core per-threat pipeline                                           #
    # ------------------------------------------------------------------ #

    def _position_at(self, s: ThreatScenario, t_ms: float) -> Tuple[float, float, float]:
        dt = (t_ms - s.appear_ms) / 1000.0
        return (
            s.start_pos_m[0] + s.velocity_mps[0] * dt,
            s.start_pos_m[1] + s.velocity_mps[1] * dt,
            s.start_pos_m[2] + s.velocity_mps[2] * dt,
        )

    def _collect_detection(
        self, s: ThreatScenario, now_ms: float
    ) -> Tuple[Optional[RadarPacket], Optional[FusedTrack], float]:
        """Sweep every radar once at ``now_ms``; return the earliest-arriving
        valid packet *and* the fused track after it is ingested."""
        pos = self._position_at(s, now_ms)
        best_pkt: Optional[RadarPacket] = None
        for node in self.radars:
            pkt = node.observe(now_ms, s.target_id, pos, s.velocity_mps, self.rng)
            if pkt is None:
                continue
            # Earliest valid arrival wins the race for confirmation slot.
            if best_pkt is None or pkt.arrive_time_ms < best_pkt.arrive_time_ms:
                best_pkt = pkt

        if best_pkt is None:
            return None, None, now_ms

        # Occlusion window: drop with some probability inside the window.
        occ = self.cfg.radars.occlusion_window_s
        if occ is not None and occ[0] * 1000.0 <= now_ms <= occ[1] * 1000.0:
            if self.rng.random() < self.cfg.radars.occlusion_fraction:
                best_pkt.valid = False

        track = self.fuser.ingest(best_pkt)
        return best_pkt, track, best_pkt.arrive_time_ms

    # Scheduler-policy multipliers applied to the contention penalty.
    # The numbers reflect the relative cost of mis-ordering jobs:
    # FIFO queues everything; EDF+slack preempts aggressively.
    _POLICY_MULT = {
        "fifo": 1.00,
        "rm": 0.75,
        "round_robin": 0.95,
        "edf": 0.55,
        "edf_slack": 0.35,
        "priority_queue": 0.45,
    }

    def _precompute_concurrency(self, scenarios: List[ThreatScenario]) -> None:
        """Count threats within the ``end_to_end`` deadline window of each
        other threat to model decision-plane contention."""
        window = self.cfg.decision.deadline_ms.end_to_end
        for s in scenarios:
            s.concurrent_siblings = sum(
                1 for o in scenarios
                if o.target_id != s.target_id
                and abs(o.appear_ms - s.appear_ms) <= window
            )

    def _contention_ms(self, s: ThreatScenario) -> float:
        base = 18.0 * getattr(s, "concurrent_siblings", 0)
        mult = self._POLICY_MULT.get(self.cfg.decision.scheduler, 1.0)
        # Prioritizer "round_robin" independently worsens tail latency.
        if self.cfg.decision.prioritizer == "round_robin":
            mult *= 1.25
        return base * mult

    def _process_threat(self, s: ThreatScenario) -> EventRecord:
        det_start_ms = s.appear_ms
        tick = 0
        pkts_confirmed = 0
        first_arrival_ms: Optional[float] = None
        confirmed_ms: Optional[float] = None
        last_track: Optional[FusedTrack] = None

        # ---- Stage 1 + 2: detection + confirmation ----
        while tick < 40:
            now_ms = det_start_ms + tick * self.cfg.radars.revisit_ms
            pkt, track, arrival = self._collect_detection(s, now_ms)
            if pkt is None or track is None or not pkt.valid:
                tick += 1
                continue
            self.confirmer.observe(s.target_id, arrival)
            last_track = track
            if first_arrival_ms is None:
                first_arrival_ms = arrival
            if self.confirmer.is_confirmed(s.target_id):
                confirmed_ms = arrival
                break
            tick += 1

        if first_arrival_ms is None or last_track is None or confirmed_ms is None:
            # Target never acquired — treat as suppressed by design.
            return EventRecord(
                target_id=s.target_id, detection_ms=det_start_ms,
                deadline_ms=det_start_ms + self.cfg.decision.deadline_ms.end_to_end,
                launched=False, hit=False, shot_down=False,
                aborted=False, abort_within_deadline=False,
                abort_reason="", return_safe=False,
                suppressed=True, suppression_reason="no_acquisition",
                deadline_met=False, end_to_end_ms=0.0,
                stage_latencies_ms={}, maneuvering=s.maneuver,
                handoff_latency_ms=last_track.handoff_latency_ms if last_track else 0.0,
            )

        detection_lat = first_arrival_ms - det_start_ms
        confirm_lat = confirmed_ms - first_arrival_ms
        # Without degraded-mode tracking, a radar dropout invalidates the
        # current track and forces an M-of-N re-acquisition; amortize the
        # expected cost into the confirmation latency.
        if not self.cfg.tracker.degraded_mode:
            confirm_lat += 25.0 * self.cfg.radars.dropout_rate * 40.0

        # ---- Stage 3: fusion (latency tied to number of contributing nodes) ----
        n_nodes = max(1, len(last_track.contributing_nodes))
        fusion_lat = 6.0 + 3.0 * n_nodes
        fusion_lat += float(self.rng.normal(0.0, 1.5))
        fusion_lat = max(2.0, fusion_lat)
        # When fusion is disabled, the downstream Kalman filter has to
        # compensate with a longer burn-in, which we model as added
        # latency proportional to the measurement noise.
        if not self.cfg.radars.fusion_enabled:
            fusion_lat += 8.0 + 0.2 * self.cfg.tracker.meas_noise_m

        # ---- Stage 4: decision (scheduler picks this job immediately) ----
        threat = score_threat(
            last_track.position_m,
            last_track.velocity_mps,
            self.cfg.city,
            target_class_conf=s.target_class_conf,
        )
        deadline_ms = det_start_ms + self.cfg.decision.deadline_ms.end_to_end
        job = self.scheduler.submit(
            target_id=s.target_id,
            created_ms=det_start_ms,
            deadline_ms=deadline_ms,
            threat_score=threat,
        )
        decide_lat = 8.0 + 4.0 * (1 - threat) + float(self.rng.normal(0.0, 1.8))
        decide_lat = max(3.0, decide_lat)
        contention = self._contention_ms(s)
        decide_lat += contention

        # ---- Stage 5: safety guard + authorization + launch gate ----
        verdict = self.safety.check(
            last_track.position_m, last_track.velocity_mps,
            threat_score=threat,
            target_class_conf=s.target_class_conf,
            authorized=True,   # human-in-the-loop draw below
        )

        if verdict.decision is SafetyDecision.ABORT:
            ab = self.abort.abort(verdict.reason, self.rng, engagement_progress=0.0)
            auth_lat = max(4.0, float(self.rng.normal(6.0, 2.0)))
            total = detection_lat + confirm_lat + fusion_lat + decide_lat + auth_lat
            return EventRecord(
                target_id=s.target_id, detection_ms=det_start_ms,
                deadline_ms=deadline_ms, launched=False, hit=False,
                shot_down=False, aborted=True,
                abort_within_deadline=ab.within_deadline,
                abort_reason=verdict.reason, return_safe=ab.return_safe,
                suppressed=False, suppression_reason="",
                deadline_met=total <= self.cfg.decision.deadline_ms.end_to_end,
                end_to_end_ms=total,
                stage_latencies_ms={
                    "detection": detection_lat, "track_confirm": confirm_lat,
                    "fusion": fusion_lat, "decision": decide_lat,
                    "authorize": auth_lat,
                },
                maneuvering=s.maneuver,
                handoff_latency_ms=last_track.handoff_latency_ms,
            )

        if verdict.decision is SafetyDecision.SUPPRESS:
            auth_lat = max(3.0, float(self.rng.normal(5.0, 1.5)))
            total = detection_lat + confirm_lat + fusion_lat + decide_lat + auth_lat
            return EventRecord(
                target_id=s.target_id, detection_ms=det_start_ms,
                deadline_ms=deadline_ms, launched=False, hit=False,
                shot_down=False, aborted=False, abort_within_deadline=False,
                abort_reason="", return_safe=False,
                suppressed=True, suppression_reason=verdict.reason,
                deadline_met=total <= self.cfg.decision.deadline_ms.end_to_end,
                end_to_end_ms=total,
                stage_latencies_ms={
                    "detection": detection_lat, "track_confirm": confirm_lat,
                    "fusion": fusion_lat, "decision": decide_lat,
                    "authorize": auth_lat,
                },
                maneuvering=s.maneuver,
                handoff_latency_ms=last_track.handoff_latency_ms,
            )

        lo = self.gate.authorize(
            threat_score=threat,
            target_class_conf=s.target_class_conf,
            safety_allow=True,
            rng=self.rng,
        )
        if not lo.launched:
            total = detection_lat + confirm_lat + fusion_lat + decide_lat + lo.authorization_ms
            return EventRecord(
                target_id=s.target_id, detection_ms=det_start_ms,
                deadline_ms=deadline_ms, launched=False, hit=False,
                shot_down=False, aborted=False,
                abort_within_deadline=False, abort_reason="",
                return_safe=False, suppressed=True,
                suppression_reason=lo.reason,
                deadline_met=total <= self.cfg.decision.deadline_ms.end_to_end,
                end_to_end_ms=total,
                stage_latencies_ms={
                    "detection": detection_lat, "track_confirm": confirm_lat,
                    "fusion": fusion_lat, "decision": decide_lat,
                    "authorize": lo.authorization_ms,
                },
                maneuvering=s.maneuver,
                handoff_latency_ms=last_track.handoff_latency_ms,
            )

        # ---- Operator abort simulates sortie 8 ----
        if s.operator_abort:
            ab = self.abort.abort("operator", self.rng, engagement_progress=0.2)
            total = (detection_lat + confirm_lat + fusion_lat + decide_lat
                     + lo.authorization_ms + ab.latency_ms)
            return EventRecord(
                target_id=s.target_id, detection_ms=det_start_ms,
                deadline_ms=deadline_ms, launched=True, hit=False,
                shot_down=False, aborted=True,
                abort_within_deadline=ab.within_deadline,
                abort_reason="operator", return_safe=ab.return_safe,
                suppressed=False, suppression_reason="",
                deadline_met=total <= self.cfg.decision.deadline_ms.end_to_end,
                end_to_end_ms=total,
                stage_latencies_ms={
                    "detection": detection_lat, "track_confirm": confirm_lat,
                    "fusion": fusion_lat, "decision": decide_lat,
                    "authorize": lo.authorization_ms,
                    "abort": ab.latency_ms,
                },
                maneuvering=s.maneuver,
                handoff_latency_ms=last_track.handoff_latency_ms,
            )

        # ---- Stage 6: launch actuation ----
        launch_lat = float(self.rng.normal(
            self.cfg.decision.deadline_ms.launch_actuation * 0.7, 18.0
        ))
        launch_lat = max(40.0, launch_lat)

        # ---- Stage 7: interceptor kinematics ----
        # Target position when the interceptor leaves the cradle.
        flight_ready_ms = det_start_ms + detection_lat + confirm_lat + fusion_lat \
            + decide_lat + lo.authorization_ms + launch_lat
        pos_now = self._position_at(s, flight_ready_ms)

        er = self.interceptor.engage(
            launch_point_km=self.cfg.interceptor.base_km,
            target_pos_m=pos_now,
            target_vel_mps=s.velocity_mps,
            target_maneuvering=s.maneuver,
            rng=self.rng,
        )

        # Require_lost: lock lost before valid geometry -> abort.
        if s.require_lost:
            ab = self.abort.abort("target_lost", self.rng, engagement_progress=0.5)
            total = (detection_lat + confirm_lat + fusion_lat + decide_lat
                     + lo.authorization_ms + launch_lat + ab.latency_ms)
            return EventRecord(
                target_id=s.target_id, detection_ms=det_start_ms,
                deadline_ms=deadline_ms, launched=True, hit=False,
                shot_down=False, aborted=True,
                abort_within_deadline=ab.within_deadline,
                abort_reason="target_lost", return_safe=ab.return_safe,
                suppressed=False, suppression_reason="",
                deadline_met=False,
                end_to_end_ms=total,
                stage_latencies_ms={
                    "detection": detection_lat, "track_confirm": confirm_lat,
                    "fusion": fusion_lat, "decision": decide_lat,
                    "authorize": lo.authorization_ms,
                    "launch_actuation": launch_lat, "abort": ab.latency_ms,
                },
                maneuvering=s.maneuver,
                handoff_latency_ms=last_track.handoff_latency_ms,
            )

        total = (detection_lat + confirm_lat + fusion_lat + decide_lat
                 + lo.authorization_ms + launch_lat + er.reaction_ms)

        deadline_met = total <= self.cfg.decision.deadline_ms.end_to_end

        return EventRecord(
            target_id=s.target_id, detection_ms=det_start_ms,
            deadline_ms=deadline_ms, launched=True, hit=er.hit,
            shot_down=er.shot_down, aborted=False,
            abort_within_deadline=False, abort_reason="",
            return_safe=False, suppressed=False, suppression_reason="",
            deadline_met=deadline_met, end_to_end_ms=total,
            stage_latencies_ms={
                "detection": detection_lat, "track_confirm": confirm_lat,
                "fusion": fusion_lat, "decision": decide_lat,
                "authorize": lo.authorization_ms,
                "launch_actuation": launch_lat,
                "interceptor_reaction": er.reaction_ms,
            },
            maneuvering=s.maneuver,
            handoff_latency_ms=last_track.handoff_latency_ms,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, scenarios: List[ThreatScenario]) -> RuntimeReport:
        self._precompute_concurrency(scenarios)
        for s in scenarios:
            rec = self._process_threat(s)
            self.metrics.events.append(rec)
            # Periodically drop stale confirmations/tracks to keep memory O(1).
            if len(self.metrics.events) % 50 == 0:
                self.confirmer.drop(s.target_id)
                self.kalman.drop(s.target_id)
                self.fuser.drop(s.target_id)

        handoffs = [e.handoff_latency_ms for e in self.metrics.events
                    if e.handoff_latency_ms > 0]
        return RuntimeReport(
            metrics=self.metrics,
            scheduler_policy=self.cfg.decision.scheduler,
            num_threats=len(scenarios),
            handoff_latencies_ms=handoffs,
        )

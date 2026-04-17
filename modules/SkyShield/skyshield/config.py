"""Typed YAML-loadable configuration dataclasses for SkyShield."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScenarioConfig:
    name: str = "urban_300km2"
    area_km2: float = 300.0
    duration_s: float = 600.0
    num_sorties: int = 50
    population_density: int = 4200
    no_fly_zones: int = 6


@dataclass
class RadarConfig:
    num_nodes: int = 4
    range_km: float = 12.0
    azimuth_dwell_ms: float = 35.0
    detection_pd_at_max: float = 0.78
    false_alarm_per_min: float = 0.6
    packet_jitter_ms_std: float = 4.0
    packet_dropout_pct: float = 0.5


@dataclass
class TrackerConfig:
    model: str = "imm_kf"
    process_noise_q: float = 1.5
    measurement_noise_r: float = 6.0
    m_of_n_m: int = 4
    m_of_n_n: int = 6
    max_track_age_ms: int = 1500


@dataclass
class FusionConfig:
    method: str = "covariance_weighted"
    handoff_overlap_m: float = 800.0
    handoff_budget_ms: float = 35.0


@dataclass
class DeadlineConfig:
    scheduler: str = "rm_edf_slack"
    end_to_end_ms: int = 1500
    detection_ms: int = 60
    track_confirm_ms: int = 80
    fusion_ms: int = 25
    decision_ms: int = 30
    launch_actuation_ms: int = 120
    interceptor_reaction_ms: int = 250
    abort_deadline_ms: int = 200
    p99_target_ms: int = 1450
    miss_target_pct: float = 0.5


@dataclass
class DecisionConfig:
    threat_score_threshold: float = 0.62
    prioritization: str = "weighted_threat"
    multi_target_max: int = 8
    authorization_check_ms_mean: float = 18.0
    authorization_check_ms_std: float = 6.0
    geofence_buffer_m: float = 200.0


@dataclass
class InterceptorConfig:
    max_speed_kmh: float = 350.0
    cruise_speed_kmh: float = 200.0
    endurance_s: float = 210.0
    hit_prob_base: float = 0.80
    return_safe: bool = True
    reload_time_s: float = 15.0
    base_lat_lon: tuple[float, float] = (0.0, 0.0)


@dataclass
class SafetyGuardConfig:
    require_authorization: bool = True
    require_friendly_airspace_clear: bool = True
    require_class_confidence: float = 0.7
    require_geofence_clear: bool = True
    abort_on_lost_lock: bool = True
    false_launch_suppression_target_pct: float = 99.5


@dataclass
class TelemetryConfig:
    trace_events: bool = True
    emit_per_sortie_record: bool = True


@dataclass
class SkyShieldConfig:
    seed: int = 20260418
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    deadline: DeadlineConfig = field(default_factory=DeadlineConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    interceptor: InterceptorConfig = field(default_factory=InterceptorConfig)
    safety_guard: SafetyGuardConfig = field(default_factory=SafetyGuardConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    output_dir: str = "outputs"

    raw: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "SkyShieldConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return SkyShieldConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "SkyShieldConfig":
        cfg = SkyShieldConfig(
            seed=int(raw.get("seed", 20260418)),
            scenario=ScenarioConfig(**raw.get("scenario", {})),
            radar=RadarConfig(**raw.get("radar", {})),
            tracker=TrackerConfig(**raw.get("tracker", {})),
            fusion=FusionConfig(**raw.get("fusion", {})),
            deadline=DeadlineConfig(**raw.get("deadline", {})),
            decision=DecisionConfig(**raw.get("decision", {})),
            interceptor=InterceptorConfig(**raw.get("interceptor", {})),
            safety_guard=SafetyGuardConfig(**raw.get("safety_guard", {})),
            telemetry=TelemetryConfig(**raw.get("telemetry", {})),
            output_dir=raw.get("output_dir", "outputs"),
            raw=raw,
        )
        return cfg

    def with_overrides(self, **overrides: Any) -> "SkyShieldConfig":
        return replace(self, **overrides)

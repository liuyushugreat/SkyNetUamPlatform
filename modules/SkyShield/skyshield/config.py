"""YAML-driven configuration for SkyShield.

The YAML is authored by the reviewer; we parse it into a nested
dataclass so the rest of the code never peeks into raw dicts.  The
parser intentionally silently ignores unknown keys — all experiment
scripts add regime-specific `override` maps that may introduce
auxiliary knobs (e.g. `radars.occlusion_window_s`), which are handled
lazily by the consumers that care.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# -------------------------------------------------------------------------- #
# Helpers                                                                    #
# -------------------------------------------------------------------------- #


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _apply_dotted_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply overrides whose keys may be dotted paths like 'radars.count'."""
    for key, value in overrides.items():
        if "." in key:
            path = key.split(".")
            cursor = base
            for p in path[:-1]:
                cursor = cursor.setdefault(p, {})
            cursor[path[-1]] = value
        else:
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                _deep_update(base[key], value)
            else:
                base[key] = value
    return base


# -------------------------------------------------------------------------- #
# Dataclasses                                                                #
# -------------------------------------------------------------------------- #


@dataclass
class NoFlyZone:
    name: str
    center_km: Tuple[float, float]
    radius_km: float


@dataclass
class CityConfig:
    name: str
    area_km2: float
    bbox_km: Tuple[float, float, float, float]
    no_fly_zones: List[NoFlyZone] = field(default_factory=list)


@dataclass
class RadarConfig:
    count: int
    coverage_km: float
    range_km_max: float
    dwell_ms: float
    revisit_ms: float
    packet_mean_ms: float
    packet_jitter_ms: float
    dropout_rate: float
    placement: List[Tuple[float, float]]
    fusion_enabled: bool = True
    occlusion_window_s: Optional[Tuple[float, float]] = None
    occlusion_fraction: float = 0.0


@dataclass
class TrackerConfig:
    process_noise: float
    meas_noise_m: float
    confirm_m_of_n: Tuple[int, int]
    gate_sigma: float
    degraded_mode: bool = True


@dataclass
class DeadlineBudget:
    detection: float
    track_confirm: float
    fusion: float
    decision: float
    launch_actuation: float
    interceptor_reaction: float
    end_to_end: float


@dataclass
class DecisionConfig:
    deadline_ms: DeadlineBudget
    scheduler: str
    threat_threshold: float
    authorization_ms_mean: float
    authorization_ms_std: float
    false_launch_block: bool
    prioritizer: str
    cmd_jitter_ms: float = 6.0


@dataclass
class SafetyConfig:
    guard_enabled: bool
    abort_deadline_ms: float
    return_safe_enabled: bool
    geofence_margin_m: float
    friendly_airspace_check: bool
    target_class_conf_min: float


@dataclass
class InterceptorConfig:
    mass_kg: float
    max_speed_mps: float
    cruise_speed_mps: float
    endurance_s: float
    acc_mps2: float
    hit_prob_nominal: float
    hit_prob_under_maneuver: float
    base_km: Tuple[float, float]


@dataclass
class WorkloadConfig:
    duration_s: float
    threat_arrival_rate_hz: float
    target_speed_mean: float
    target_speed_std: float
    target_altitude_mean: float
    target_altitude_std: float
    maneuver_prob: float
    authorized_loss_budget: float


@dataclass
class SkyShieldConfig:
    seed: int
    city: CityConfig
    radars: RadarConfig
    tracker: TrackerConfig
    decision: DecisionConfig
    safety: SafetyConfig
    interceptor: InterceptorConfig
    workload: WorkloadConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, overrides: Dict[str, Any]) -> "SkyShieldConfig":
        raw = yaml.safe_load(yaml.safe_dump(self.raw))  # deep copy via yaml roundtrip
        _apply_dotted_overrides(raw, overrides)
        return _build_config(raw)


# -------------------------------------------------------------------------- #
# Builders                                                                   #
# -------------------------------------------------------------------------- #


def _build_config(raw: Dict[str, Any]) -> SkyShieldConfig:
    city_cfg = raw["city"]
    nfz = [NoFlyZone(**z) for z in city_cfg.get("no_fly_zones", [])]
    city = CityConfig(
        name=city_cfg["name"],
        area_km2=float(city_cfg["area_km2"]),
        bbox_km=tuple(city_cfg["bbox_km"]),
        no_fly_zones=nfz,
    )

    rcfg = raw["radars"]
    radars = RadarConfig(
        count=int(rcfg["count"]),
        coverage_km=float(rcfg["coverage_km"]),
        range_km_max=float(rcfg["range_km_max"]),
        dwell_ms=float(rcfg["dwell_ms"]),
        revisit_ms=float(rcfg["revisit_ms"]),
        packet_mean_ms=float(rcfg["packet_mean_ms"]),
        packet_jitter_ms=float(rcfg["packet_jitter_ms"]),
        dropout_rate=float(rcfg["dropout_rate"]),
        placement=[tuple(p) for p in rcfg["placement"]],
        fusion_enabled=bool(rcfg.get("fusion_enabled", True)),
        occlusion_window_s=(
            tuple(rcfg["occlusion_window_s"]) if rcfg.get("occlusion_window_s") else None
        ),
        occlusion_fraction=float(rcfg.get("occlusion_fraction", 0.0)),
    )

    tcfg = raw["tracker"]
    tracker = TrackerConfig(
        process_noise=float(tcfg["process_noise"]),
        meas_noise_m=float(tcfg["meas_noise_m"]),
        confirm_m_of_n=tuple(tcfg["confirm_m_of_n"]),
        gate_sigma=float(tcfg["gate_sigma"]),
        degraded_mode=bool(tcfg.get("degraded_mode", True)),
    )

    dcfg = raw["decision"]
    dlb = dcfg["deadline_ms"]
    decision = DecisionConfig(
        deadline_ms=DeadlineBudget(
            detection=float(dlb["detection"]),
            track_confirm=float(dlb["track_confirm"]),
            fusion=float(dlb["fusion"]),
            decision=float(dlb["decision"]),
            launch_actuation=float(dlb["launch_actuation"]),
            interceptor_reaction=float(dlb["interceptor_reaction"]),
            end_to_end=float(dlb["end_to_end"]),
        ),
        scheduler=str(dcfg["scheduler"]),
        threat_threshold=float(dcfg["threat_threshold"]),
        authorization_ms_mean=float(dcfg["authorization_ms_mean"]),
        authorization_ms_std=float(dcfg["authorization_ms_std"]),
        false_launch_block=bool(dcfg["false_launch_block"]),
        prioritizer=str(dcfg["prioritizer"]),
        cmd_jitter_ms=float(dcfg.get("cmd_jitter_ms", 6.0)),
    )

    scfg = raw["safety"]
    safety = SafetyConfig(
        guard_enabled=bool(scfg["guard_enabled"]),
        abort_deadline_ms=float(scfg["abort_deadline_ms"]),
        return_safe_enabled=bool(scfg["return_safe_enabled"]),
        geofence_margin_m=float(scfg["geofence_margin_m"]),
        friendly_airspace_check=bool(scfg["friendly_airspace_check"]),
        target_class_conf_min=float(scfg["target_class_conf_min"]),
    )

    icfg = raw["interceptor"]
    interceptor = InterceptorConfig(
        mass_kg=float(icfg["mass_kg"]),
        max_speed_mps=float(icfg["max_speed_mps"]),
        cruise_speed_mps=float(icfg["cruise_speed_mps"]),
        endurance_s=float(icfg["endurance_s"]),
        acc_mps2=float(icfg["acc_mps2"]),
        hit_prob_nominal=float(icfg["hit_prob_nominal"]),
        hit_prob_under_maneuver=float(icfg["hit_prob_under_maneuver"]),
        base_km=tuple(icfg["base_km"]),
    )

    wcfg = raw["workload"]
    workload = WorkloadConfig(
        duration_s=float(wcfg["duration_s"]),
        threat_arrival_rate_hz=float(wcfg["threat_arrival_rate_hz"]),
        target_speed_mean=float(wcfg["target_speed_mps"]["mean"]),
        target_speed_std=float(wcfg["target_speed_mps"]["std"]),
        target_altitude_mean=float(wcfg["target_altitude_m"]["mean"]),
        target_altitude_std=float(wcfg["target_altitude_m"]["std"]),
        maneuver_prob=float(wcfg["maneuver_prob"]),
        authorized_loss_budget=float(wcfg["authorized_loss_budget"]),
    )

    return SkyShieldConfig(
        seed=int(raw["seed"]),
        city=city,
        radars=radars,
        tracker=tracker,
        decision=decision,
        safety=safety,
        interceptor=interceptor,
        workload=workload,
        raw=raw,
    )


def load_config(path: str | Path) -> SkyShieldConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "base" in raw:
        base_path = path.parent / raw["base"]
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        # Only the literal defaults pour into the SkyShieldConfig; the
        # experiment-specific keys (`sweep`, `variants`, `regimes`) are
        # stashed under `raw` for the scripts to inspect.
        raw = {**base, **{k: v for k, v in raw.items() if k != "base"}}
    return _build_config(raw)

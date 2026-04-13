"""Metric calculation helpers for valuation dimensions.

Each function takes a :class:`FlightAssetUnit` (with attached evidence) and
returns a score in [0, 1].
"""

from __future__ import annotations

from ..models.asset_unit import FlightAssetUnit


def completeness(unit: FlightAssetUnit) -> float:
    """How complete is the telemetry / metadata package?"""
    ev = unit.evidence
    if ev is None:
        return 0.0
    score = 0.0
    ts = ev.telemetry_summary
    if ts.total_points > 0:
        score += 0.3
    if ts.avg_altitude_m > 0:
        score += 0.1
    if ts.trajectory_length_m > 0:
        score += 0.1
    if ev.raw_data_hash:
        score += 0.2
    if ev.trajectory_hash:
        score += 0.1
    if ev.mission_result.completed:
        score += 0.2
    return min(score, 1.0)


def temporal_continuity(unit: FlightAssetUnit) -> float:
    """Approximation of time-series continuity based on point density."""
    ev = unit.evidence
    if ev is None or ev.duration_seconds <= 0:
        return 0.0
    density = ev.telemetry_summary.total_points / ev.duration_seconds
    if density >= 1.0:
        return 1.0
    if density >= 0.5:
        return 0.8
    if density >= 0.1:
        return 0.5
    return 0.2


def sensor_reliability(unit: FlightAssetUnit) -> float:
    """Heuristic sensor confidence based on anomaly / alert counts."""
    ev = unit.evidence
    if ev is None:
        return 0.0
    mr = ev.mission_result
    penalty = len(mr.anomalies) * 0.15 + len(mr.alerts) * 0.05
    return max(1.0 - penalty, 0.0)


def event_richness(unit: FlightAssetUnit) -> float:
    """More operational events → richer dataset for ML / analytics."""
    ev = unit.evidence
    if ev is None:
        return 0.0
    events = (
        len(ev.environment.risk_events)
        + len(ev.mission_result.anomalies)
        + len(ev.mission_result.alerts)
        + (1 if ev.environment.no_fly_zone_incursions > 0 else 0)
    )
    if events >= 5:
        return 1.0
    return events / 5.0


def compliance_degree(unit: FlightAssetUnit) -> float:
    """Directly proxied from the governance-assigned compliance score."""
    return max(min(unit.compliance_score, 1.0), 0.0)


def scarcity(unit: FlightAssetUnit) -> float:
    """Placeholder — in production this would query a catalogue index."""
    mt = (unit.mission_type or "").lower()
    rare_keywords = {"emergency", "night", "beyond_vlos", "urban", "weather"}
    hits = sum(1 for kw in rare_keywords if kw in mt)
    return min(hits * 0.25, 1.0)


def scenario_relevance(unit: FlightAssetUnit, target_scenario: str = "") -> float:
    """Basic keyword overlap between mission type and a target scenario."""
    if not target_scenario:
        return 0.5
    mt = (unit.mission_type or "").lower()
    if target_scenario.lower() in mt:
        return 1.0
    return 0.3


def reuse_potential(unit: FlightAssetUnit) -> float:
    """Higher quality + richer events → more reuse scenarios."""
    c = completeness(unit)
    e = event_richness(unit)
    s = sensor_reliability(unit)
    return 0.4 * c + 0.3 * e + 0.3 * s


def timeliness(unit: FlightAssetUnit) -> float:
    """Fresher data is more valuable.  Placeholder returns full score."""
    return 1.0

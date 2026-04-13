"""Metric calculation helpers for valuation dimensions.

Each function takes a :class:`FlightAssetUnit` (with attached evidence) and
returns a score in **[0, 1]**.

Scoring rules are intentionally **transparent and explainable** — every
function documents exactly how its score is computed so that auditors can
reproduce the result.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..models.asset_unit import FlightAssetUnit


# ── Data Quality dimensions ─────────────────────────────────────────────────

def completeness(unit: FlightAssetUnit) -> float:
    """How complete is the telemetry / metadata package?

    Scoring breakdown (additive, max 1.0):
    - telemetry points present          +0.30
    - avg altitude recorded             +0.10
    - trajectory length recorded        +0.10
    - raw data hash present             +0.20
    - trajectory hash present           +0.10
    - mission completed                 +0.20
    """
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
    """Time-series continuity approximated by telemetry point density.

    density = total_points / duration_seconds
    - density >= 1.0 Hz  -> 1.0  (excellent)
    - density >= 0.5 Hz  -> 0.8
    - density >= 0.1 Hz  -> 0.5
    - density <  0.1 Hz  -> 0.2  (sparse)
    """
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
    """Heuristic sensor confidence.

    Starts at 1.0, penalised by:
    - each anomaly     -0.15
    - each alert       -0.05
    Floored at 0.0.
    """
    ev = unit.evidence
    if ev is None:
        return 0.0
    mr = ev.mission_result
    penalty = len(mr.anomalies) * 0.15 + len(mr.alerts) * 0.05
    return max(1.0 - penalty, 0.0)


def event_richness(unit: FlightAssetUnit) -> float:
    """More operational events -> richer dataset for ML / analytics.

    Total events = risk_events + anomalies + alerts + nfz_incursion_flag
    - >= 5 events -> 1.0
    - otherwise   -> events / 5.0
    """
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


# ── Asset Value dimensions ──────────────────────────────────────────────────

def scarcity(unit: FlightAssetUnit) -> float:
    """Keyword-based scarcity heuristic.

    Checks mission_type against a set of rare-scenario keywords.
    Each match adds 0.25 (max 1.0).

    Keywords: emergency, night, beyond_vlos, urban, weather

    TODO(catalogue): in production, query a data-catalogue index to
    compute real scarcity based on existing asset inventory.
    """
    mt = (unit.mission_type or "").lower()
    rare_keywords = {"emergency", "night", "beyond_vlos", "urban", "weather"}
    hits = sum(1 for kw in rare_keywords if kw in mt)
    return min(hits * 0.25, 1.0)


def scenario_relevance(unit: FlightAssetUnit, target_scenario: str = "") -> float:
    """Keyword overlap between mission type and a target scenario.

    - exact substring match   -> 1.0
    - no target provided      -> 0.5 (neutral)
    - no match                -> 0.3

    TODO(registry): accept a list of active demand scenarios from an
    external registry to compute real relevance.
    """
    if not target_scenario:
        return 0.5
    mt = (unit.mission_type or "").lower()
    if target_scenario.lower() in mt:
        return 1.0
    return 0.3


def reuse_potential(unit: FlightAssetUnit) -> float:
    """Composite: higher quality + richer events -> more reuse scenarios.

    Formula: 0.4 * completeness + 0.3 * event_richness + 0.3 * sensor_reliability
    """
    c = completeness(unit)
    e = event_richness(unit)
    s = sensor_reliability(unit)
    return 0.4 * c + 0.3 * e + 0.3 * s


def timeliness(unit: FlightAssetUnit) -> float:
    """Fresher data is more valuable.

    - age <= 1 day   -> 1.0
    - age <= 7 days  -> 0.8
    - age <= 30 days -> 0.5
    - age <= 90 days -> 0.3
    - older          -> 0.1
    """
    if unit.end_time is None:
        return 0.5
    now = datetime.now(UTC)
    end = unit.end_time
    if end.tzinfo is None:
        from datetime import timezone
        end = end.replace(tzinfo=timezone.utc)
    age_days = (now - end).total_seconds() / 86400
    if age_days <= 1:
        return 1.0
    if age_days <= 7:
        return 0.8
    if age_days <= 30:
        return 0.5
    if age_days <= 90:
        return 0.3
    return 0.1

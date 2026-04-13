"""Flight data ingestion — the entry point of the asset pipeline.

:class:`FlightIngestor` accepts a :class:`FlightIngestRecord` (raw flight
metadata + optional data URIs) and produces a :class:`FlightAssetUnit` in
``INGESTED`` status, ready for downstream provenance / governance / valuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetClass, AssetStatus


@dataclass
class FlightIngestRecord:
    """Raw input for one flight — filled by the upstream flight-management system."""

    flight_id: str
    uav_id: str
    mission_id: str = ""
    operator_id: str = ""
    mission_type: str = ""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # spatial
    waypoints: List[Dict[str, float]] = field(default_factory=list)
    corridor_id: str = ""

    # telemetry summary (lightweight — not the full stream)
    telemetry_points: int = 0
    avg_altitude_m: float = 0.0
    max_altitude_m: float = 0.0
    max_speed_mps: float = 0.0
    avg_speed_mps: float = 0.0
    min_battery_pct: float = 100.0
    avg_battery_pct: float = 100.0
    payload_active: bool = False
    trajectory_length_m: float = 0.0

    # environment
    weather_condition: str = "unknown"
    wind_speed_mps: float = 0.0
    visibility_km: float = 10.0
    temperature_c: Optional[float] = None
    no_fly_zone_incursions: int = 0
    temporary_restrictions: List[str] = field(default_factory=list)
    risk_events: List[str] = field(default_factory=list)

    # mission result
    mission_completed: bool = False
    completion_pct: float = 0.0
    deviation_m: float = 0.0
    anomalies: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    # raw data references
    raw_data_uri: str = ""
    raw_data_hash: str = ""
    trajectory_hash: str = ""

    extra: Dict[str, Any] = field(default_factory=dict)


class FlightIngestor:
    """Transforms a :class:`FlightIngestRecord` into an initial :class:`FlightAssetUnit`."""

    def ingest(self, record: FlightIngestRecord) -> FlightAssetUnit:
        duration = 0.0
        if record.start_time and record.end_time:
            duration = (record.end_time - record.start_time).total_seconds()

        risk_score = self._estimate_risk(record)

        return FlightAssetUnit(
            flight_id=record.flight_id,
            uav_id=record.uav_id,
            mission_type=record.mission_type,
            start_time=record.start_time,
            end_time=record.end_time,
            telemetry_hash=record.trajectory_hash,
            evidence_uri=record.raw_data_uri,
            risk_score=risk_score,
            asset_class=AssetClass.FLIGHT_EVIDENCE,
            status=AssetStatus.INGESTED,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_risk(record: FlightIngestRecord) -> float:
        """Quick heuristic risk score ∈ [0, 1]."""
        penalties: List[float] = []
        if record.no_fly_zone_incursions > 0:
            penalties.append(min(record.no_fly_zone_incursions * 0.2, 0.6))
        if record.violations:
            penalties.append(min(len(record.violations) * 0.15, 0.5))
        if record.anomalies:
            penalties.append(min(len(record.anomalies) * 0.1, 0.3))
        if record.deviation_m > 500:
            penalties.append(0.15)
        if not record.mission_completed:
            penalties.append(0.1)
        return min(sum(penalties), 1.0)

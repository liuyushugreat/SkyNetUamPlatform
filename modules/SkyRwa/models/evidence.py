"""Flight evidence packaging models.

A :class:`FlightEvidencePackage` is the *attestation record* for a single
flight.  It captures summary statistics, environmental context, mission
outcomes, and cryptographic references to the raw data — but it does **not**
contain the raw telemetry itself.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TelemetrySummary(BaseModel):
    """Condensed telemetry statistics for one flight."""
    total_points: int = 0
    avg_altitude_m: float = 0.0
    max_altitude_m: float = 0.0
    max_speed_mps: float = 0.0
    avg_speed_mps: float = 0.0
    min_battery_pct: float = 100.0
    avg_battery_pct: float = 100.0
    payload_active: bool = False
    trajectory_length_m: float = 0.0


class EnvironmentContext(BaseModel):
    """Ambient conditions during the flight."""
    weather_condition: str = "unknown"
    wind_speed_mps: float = 0.0
    visibility_km: float = 10.0
    temperature_c: Optional[float] = None
    no_fly_zone_incursions: int = 0
    temporary_restrictions: List[str] = Field(default_factory=list)
    risk_events: List[str] = Field(default_factory=list)


class MissionResult(BaseModel):
    """Outcome summary for the flight mission."""
    completed: bool = False
    completion_pct: float = 0.0
    deviation_m: float = 0.0
    anomalies: List[str] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)


class FlightEvidencePackage(BaseModel):
    """
    Verifiable evidence package for a single flight.

    This is **not** an asset — it is the raw attestation material from which
    governed data products (the real assets) are derived.
    """
    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    flight_id: str
    uav_id: str
    mission_id: str = ""
    operator_id: str = ""

    start_time: datetime
    end_time: datetime
    duration_seconds: float = 0.0

    trajectory_hash: str = ""
    raw_data_uri: str = ""
    raw_data_hash: str = ""

    telemetry_summary: TelemetrySummary = Field(default_factory=TelemetrySummary)
    environment: EnvironmentContext = Field(default_factory=EnvironmentContext)
    mission_result: MissionResult = Field(default_factory=MissionResult)

    digest_hash: str = ""
    signature: Optional[str] = None
    signed_by: Optional[str] = None
    signed_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = Field(default_factory=dict)

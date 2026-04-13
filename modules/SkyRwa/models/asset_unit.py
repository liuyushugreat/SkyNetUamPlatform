"""The top-level aggregate: FlightAssetUnit.

A :class:`FlightAssetUnit` is the *candidate* data-asset that a single flight
produces after passing through the ingest → provenance → governance →
valuation pipeline.  It is **not** a token or a tradable product by itself;
it becomes one only after explicit promotion through governance and
settlement steps.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from .enums import AssetClass, AssetStatus
from .evidence import FlightEvidencePackage
from .rights import RightsProfile
from .settlement import RevenueLog, SettlementRule
from .valuation import ValuationResultV2


class FlightAssetUnit(BaseModel):
    """Core aggregate representing one flight's data-asset candidacy."""

    asset_unit_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    flight_id: str
    uav_id: str
    mission_type: str = ""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    telemetry_hash: str = ""
    evidence_uri: str = ""

    compliance_score: float = 0.0
    risk_score: float = 0.0
    data_quality_score: float = 0.0

    rights_profile: Optional[RightsProfile] = None
    asset_class: AssetClass = AssetClass.FLIGHT_EVIDENCE
    valuation_result: Optional[ValuationResultV2] = None
    settlement_rule: Optional[SettlementRule] = None
    revenue_log: List[RevenueLog] = Field(default_factory=list)

    status: AssetStatus = AssetStatus.INGESTED

    evidence: Optional[FlightEvidencePackage] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

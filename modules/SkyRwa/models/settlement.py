"""Settlement rules and revenue log models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import SettlementStatus, UsageType


class SplitEntry(BaseModel):
    """One line in a revenue-split table."""
    party_id: str
    role: str
    share_pct: float
    amount: float = 0.0


class SettlementRule(BaseModel):
    """
    Defines *when* revenue is recognized and *how* it is divided among
    participants for a given asset unit.
    """
    trigger_types: List[UsageType] = Field(
        default_factory=lambda: [UsageType.API_CALL],
    )
    participants: List[SplitEntry] = Field(default_factory=list)
    min_settlement_unit: float = 0.01
    settlement_cycle_days: Optional[int] = None
    currency: str = "USD"


class RevenueLog(BaseModel):
    """Immutable record of a single revenue-generating event."""
    usage_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    asset_unit_id: str
    usage_type: UsageType = UsageType.API_CALL
    consumer: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    gross_amount: float = 0.0
    split_detail: List[SplitEntry] = Field(default_factory=list)
    settlement_status: SettlementStatus = SettlementStatus.PENDING
    metadata: Dict[str, str] = Field(default_factory=dict)

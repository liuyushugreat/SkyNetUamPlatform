"""Settlement rules, revenue logs and settlement record models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
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
    """Defines *when* revenue is recognized and *how* it is divided.

    ``participants`` carries the split ratios; these are read at settlement
    time by :class:`~SkyRwa.settlement.splitter.RevenueSplitter` —
    they are never hard-coded in business logic.
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    gross_amount: float = 0.0
    split_detail: List[SplitEntry] = Field(default_factory=list)
    settlement_status: SettlementStatus = SettlementStatus.PENDING
    metadata: Dict[str, str] = Field(default_factory=dict)


class SettlementRecord(BaseModel):
    """Finalized settlement snapshot persisted to disk.

    A ``SettlementRecord`` is created when one or more :class:`RevenueLog`
    entries are marked ``SETTLED``.  It aggregates the split amounts per
    participant and records a settlement timestamp.
    """
    settlement_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    asset_unit_id: str
    settled_usage_ids: List[str] = Field(default_factory=list)
    total_gross: float = 0.0
    participant_totals: List[SplitEntry] = Field(default_factory=list)
    currency: str = "USD"
    settled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""

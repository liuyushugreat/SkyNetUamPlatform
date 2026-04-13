"""Rights, governance and data-use policy models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .enums import DataCategory, UsageLevel


class RetentionPolicy(BaseModel):
    """How long a data asset may be kept and what happens at expiry."""
    max_retention_days: Optional[int] = None
    expires_at: Optional[str] = None
    auto_archive: bool = True


class RevenueParticipant(BaseModel):
    """One party entitled to a share of revenue from an asset."""
    party_id: str
    role: str  # owner | operator | platform | data_processor
    share_pct: float  # 0–100


class RightsProfile(BaseModel):
    """
    Describes *who* owns / controls a data asset, *what* may be done with it,
    and *how* revenue is split among participants.
    """
    owner: str
    controller: str = ""
    contributors: List[str] = Field(default_factory=list)

    data_categories: List[DataCategory] = Field(
        default_factory=lambda: [DataCategory.RAW_TELEMETRY],
    )
    permitted_uses: List[UsageLevel] = Field(
        default_factory=lambda: [UsageLevel.INTERNAL_ONLY],
    )
    prohibited_uses: List[str] = Field(default_factory=list)

    desensitization_required: bool = False
    aggregation_min_count: int = 1
    tradable: bool = False

    retention: RetentionPolicy = Field(default_factory=RetentionPolicy)
    revenue_split: List[RevenueParticipant] = Field(default_factory=list)

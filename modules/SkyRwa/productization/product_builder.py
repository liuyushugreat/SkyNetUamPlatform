"""Build GovernedDataProduct from aggregated asset candidates.

A GovernedProduct represents a curated, multi-flight data product that
has passed governance checks and is ready for licensing or trading.

Lifecycle:
    FlightEvidence → AssetCandidate → GovernedDataProduct → RevenueRight
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetClass, UsageLevel
from ..models.rights import RightsProfile, RevenueParticipant
from .aggregator import AggregationGroup


class GovernedProduct(BaseModel):
    """A governed, aggregated data product."""

    product_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    product_category: AssetClass = AssetClass.RISK_DATASET
    source_asset_ids: List[str] = Field(default_factory=list)
    source_flight_ids: List[str] = Field(default_factory=list)

    rights_summary: Optional[RightsProfile] = None
    tradable: bool = False
    suggested_value: float = 0.0
    avg_quality: float = 0.0
    avg_compliance: float = 0.0

    lineage_note: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = "created"


class ProductBuilder:
    """Construct GovernedProduct instances from aggregation groups."""

    def build(self, group: AggregationGroup) -> GovernedProduct:
        """Create a governed product from an aggregation group.

        Raises ``ValueError`` if the group has fewer than 2 candidates.
        """
        if group.count < 2:
            raise ValueError(
                f"Need at least 2 candidates to form a product, got {group.count}"
            )
        source_ids = [u.asset_unit_id for u in group.candidates]
        flight_ids = [u.flight_id for u in group.candidates]

        merged_participants = self._merge_participants(group.candidates)
        rights = RightsProfile(
            owner="platform",
            controller="platform",
            contributors=list({u.uav_id for u in group.candidates}),
            permitted_uses=[UsageLevel.LICENSED_EXTERNAL],
            tradable=True,
            desensitization_required=any(
                u.rights_profile.desensitization_required
                for u in group.candidates
                if u.rights_profile
            ),
            revenue_split=merged_participants,
        )

        avg_val = 0.0
        valued = [u for u in group.candidates if u.valuation_result]
        if valued:
            avg_val = sum(u.valuation_result.estimated_value for u in valued) / len(valued)
        product_value = avg_val * group.count * 0.8

        return GovernedProduct(
            product_category=group.asset_class,
            source_asset_ids=source_ids,
            source_flight_ids=flight_ids,
            rights_summary=rights,
            tradable=True,
            suggested_value=round(product_value, 2),
            avg_quality=round(group.avg_quality, 4),
            avg_compliance=round(group.avg_compliance, 4),
            lineage_note=f"Aggregated from {group.count} candidates of class {group.asset_class.value}",
            status="governed",
        )

    @staticmethod
    def _merge_participants(units: List[FlightAssetUnit]) -> List[RevenueParticipant]:
        """Deduplicate and average revenue participants across sources."""
        totals: dict[str, dict] = {}
        count = 0
        for u in units:
            if u.rights_profile is None:
                continue
            count += 1
            for p in u.rights_profile.revenue_split:
                key = (p.party_id, p.role)
                if key not in totals:
                    totals[key] = {"party_id": p.party_id, "role": p.role, "share_sum": 0.0}
                totals[key]["share_sum"] += p.share_pct
        if count == 0:
            return []
        return [
            RevenueParticipant(
                party_id=v["party_id"],
                role=v["role"],
                share_pct=round(v["share_sum"] / count, 2),
            )
            for v in totals.values()
        ]

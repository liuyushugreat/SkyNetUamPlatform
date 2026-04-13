"""Aggregate multiple FlightAssetUnit candidates by asset class.

The aggregator groups candidates by ``asset_class`` and filters those
meeting minimum quality / compliance thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetClass


@dataclass
class AggregationGroup:
    asset_class: AssetClass
    candidates: List[FlightAssetUnit] = field(default_factory=list)
    avg_quality: float = 0.0
    avg_compliance: float = 0.0

    @property
    def count(self) -> int:
        return len(self.candidates)


class CandidateAggregator:
    """Group and filter asset candidates for productization."""

    def __init__(
        self,
        min_quality: float = 0.5,
        min_compliance: float = 0.7,
        min_count: int = 3,
    ):
        self.min_quality = min_quality
        self.min_compliance = min_compliance
        self.min_count = min_count

    def group(self, candidates: List[FlightAssetUnit]) -> Dict[AssetClass, AggregationGroup]:
        """Group candidates by asset class, filtering by quality thresholds."""
        groups: Dict[AssetClass, List[FlightAssetUnit]] = {}
        for c in candidates:
            if c.data_quality_score < self.min_quality:
                continue
            if c.compliance_score < self.min_compliance:
                continue
            if c.rights_profile is None or not c.rights_profile.tradable:
                continue
            groups.setdefault(c.asset_class, []).append(c)

        result: Dict[AssetClass, AggregationGroup] = {}
        for cls, units in groups.items():
            if len(units) < self.min_count:
                continue
            avg_q = sum(u.data_quality_score for u in units) / len(units)
            avg_c = sum(u.compliance_score for u in units) / len(units)
            result[cls] = AggregationGroup(
                asset_class=cls,
                candidates=units,
                avg_quality=avg_q,
                avg_compliance=avg_c,
            )
        return result

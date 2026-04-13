"""Rule-based valuation engine — the default, production-ready implementation.

Scoring pipeline:
1. Compute :class:`DataQualityScore` from evidence-derived metrics.
2. Compute :class:`AssetValueScore` from business-value metrics.
3. Combine into a monetary :class:`ValuationResultV2` using configurable
   weights and a base-price multiplier.
"""

from __future__ import annotations

from datetime import datetime

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetStatus
from ..models.valuation import AssetValueScore, DataQualityScore, ValuationResultV2
from . import metrics as M
from .base import AbstractAssetValuationEngine


class RuleBasedValuationEngine(AbstractAssetValuationEngine):
    """Deterministic, configurable valuation engine."""

    engine_id: str = "rule_based"

    def __init__(
        self,
        base_price: float = 100.0,
        quality_weight: float = 0.6,
        value_weight: float = 0.4,
        currency: str = "USD",
    ):
        self.base_price = base_price
        self.quality_weight = quality_weight
        self.value_weight = value_weight
        self.currency = currency

    def evaluate(self, unit: FlightAssetUnit) -> ValuationResultV2:
        qs = self._quality_score(unit)
        vs = self._value_score(unit)

        combined = self.quality_weight * qs.overall + self.value_weight * vs.overall
        estimated = round(self.base_price * combined, 4)
        confidence = round(min(qs.overall, vs.overall), 4)

        result = ValuationResultV2(
            asset_unit_id=unit.asset_unit_id,
            quality_score=qs,
            value_score=vs,
            estimated_value=estimated,
            currency=self.currency,
            confidence=confidence,
            breakdown={
                "base_price": self.base_price,
                "quality_weight": self.quality_weight,
                "value_weight": self.value_weight,
                "quality_overall": qs.overall,
                "value_overall": vs.overall,
                "combined_factor": combined,
            },
            engine_id=self.engine_id,
        )

        unit.valuation_result = result
        unit.data_quality_score = qs.overall
        unit.status = AssetStatus.VALUATED
        unit.updated_at = datetime.utcnow()
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _quality_score(unit: FlightAssetUnit) -> DataQualityScore:
        c = M.completeness(unit)
        t = M.temporal_continuity(unit)
        s = M.sensor_reliability(unit)
        e = M.event_richness(unit)
        comp = M.compliance_degree(unit)
        overall = round(0.25 * c + 0.20 * t + 0.20 * s + 0.15 * e + 0.20 * comp, 4)
        return DataQualityScore(
            completeness=round(c, 4),
            temporal_continuity=round(t, 4),
            sensor_reliability=round(s, 4),
            event_richness=round(e, 4),
            compliance_degree=round(comp, 4),
            overall=overall,
        )

    @staticmethod
    def _value_score(unit: FlightAssetUnit) -> AssetValueScore:
        sc = M.scarcity(unit)
        sr = M.scenario_relevance(unit)
        rp = M.reuse_potential(unit)
        tl = M.timeliness(unit)
        overall = round(0.25 * sc + 0.25 * sr + 0.30 * rp + 0.20 * tl, 4)
        return AssetValueScore(
            scarcity=round(sc, 4),
            scenario_relevance=round(sr, 4),
            reuse_potential=round(rp, 4),
            timeliness=round(tl, 4),
            overall=overall,
        )

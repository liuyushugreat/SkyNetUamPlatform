"""Rule-based valuation engine — the default, production-ready implementation.

Scoring pipeline (fully transparent, no black-box)
---------------------------------------------------
1. **DataQualityScore** — intrinsic data quality from evidence metrics:
   - completeness        (weight 0.25)
   - temporal_continuity (weight 0.20)
   - sensor_reliability  (weight 0.20)
   - event_richness      (weight 0.15)
   - compliance_degree   (weight 0.20)

2. **AssetValueScore** — extrinsic business-value metrics:
   - scarcity            (weight 0.25)
   - scenario_relevance  (weight 0.25)
   - reuse_potential     (weight 0.30)
   - timeliness          (weight 0.20)

3. **Estimated value** = ``base_price * (quality_weight * Q + value_weight * V)``

Higher completeness + compliance + event richness + scarcity = bonus.
Missing data + violations + non-reusable = penalty.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetStatus
from ..models.valuation import AssetValueScore, DataQualityScore, ValuationResultV2
from . import metrics as M
from .base import AbstractAssetValuationEngine

# Weights are module-level constants so they can be inspected / overridden.
_QUALITY_WEIGHTS = {
    "completeness": 0.25,
    "temporal_continuity": 0.20,
    "sensor_reliability": 0.20,
    "event_richness": 0.15,
    "compliance_degree": 0.20,
}

_VALUE_WEIGHTS = {
    "scarcity": 0.25,
    "scenario_relevance": 0.25,
    "reuse_potential": 0.30,
    "timeliness": 0.20,
}


class RuleBasedValuationEngine(AbstractAssetValuationEngine):
    """Deterministic, configurable, explainable valuation engine.

    Raises
    ------
    ValueError
        If *unit* has no attached evidence.
    """

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
        if unit.evidence is None:
            raise ValueError(
                f"Cannot valuate asset unit {unit.asset_unit_id}: "
                "evidence package is missing"
            )

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
                "combined_factor": round(combined, 4),
            },
            engine_id=self.engine_id,
            notes=(
                "estimated_value = base_price * "
                "(quality_weight * quality_overall + value_weight * value_overall)"
            ),
        )

        unit.valuation_result = result
        unit.data_quality_score = qs.overall
        unit.status = AssetStatus.VALUATED
        unit.updated_at = datetime.now(UTC)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _quality_score(unit: FlightAssetUnit) -> DataQualityScore:
        w = _QUALITY_WEIGHTS
        c = M.completeness(unit)
        t = M.temporal_continuity(unit)
        s = M.sensor_reliability(unit)
        e = M.event_richness(unit)
        comp = M.compliance_degree(unit)
        overall = round(
            w["completeness"] * c
            + w["temporal_continuity"] * t
            + w["sensor_reliability"] * s
            + w["event_richness"] * e
            + w["compliance_degree"] * comp,
            4,
        )
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
        w = _VALUE_WEIGHTS
        sc = M.scarcity(unit)
        sr = M.scenario_relevance(unit)
        rp = M.reuse_potential(unit)
        tl = M.timeliness(unit)
        overall = round(
            w["scarcity"] * sc
            + w["scenario_relevance"] * sr
            + w["reuse_potential"] * rp
            + w["timeliness"] * tl,
            4,
        )
        return AssetValueScore(
            scarcity=round(sc, 4),
            scenario_relevance=round(sr, 4),
            reuse_potential=round(rp, 4),
            timeliness=round(tl, 4),
            overall=overall,
        )

"""Product-level valuation: estimate value for GovernedDataProduct.

Extends candidate-level valuation to aggregated products, accounting
for volume, diversity, and governance-imposed constraints.
"""

from __future__ import annotations

from typing import Optional

from ..productization.product_builder import GovernedProduct
from .explanation import ValuationExplanation, ValuationFactor, GovernanceImpactOnValue


class ProductValuationEngine:
    """Estimate the value of a GovernedDataProduct."""

    VOLUME_WEIGHT = 0.30
    DIVERSITY_WEIGHT = 0.20
    QUALITY_WEIGHT = 0.30
    COMPLIANCE_WEIGHT = 0.20
    BASE_UNIT_VALUE = 50.0

    def valuate(self, product: GovernedProduct) -> ValuationExplanation:
        n = len(product.source_asset_ids)
        unique_flights = len(set(product.source_flight_ids))
        diversity = unique_flights / max(n, 1)

        volume_score = min(n / 10.0, 1.0)
        quality_score = product.avg_quality
        compliance_score = product.avg_compliance

        factors = [
            ValuationFactor(
                name="volume",
                raw_score=volume_score,
                weight=self.VOLUME_WEIGHT,
                weighted_score=volume_score * self.VOLUME_WEIGHT,
                impact="positive" if volume_score >= 0.5 else "neutral",
                detail=f"{n} source assets",
            ),
            ValuationFactor(
                name="diversity",
                raw_score=diversity,
                weight=self.DIVERSITY_WEIGHT,
                weighted_score=diversity * self.DIVERSITY_WEIGHT,
                impact="positive" if diversity >= 0.8 else "neutral",
                detail=f"{unique_flights} unique flights out of {n} assets",
            ),
            ValuationFactor(
                name="avg_quality",
                raw_score=quality_score,
                weight=self.QUALITY_WEIGHT,
                weighted_score=quality_score * self.QUALITY_WEIGHT,
                impact="positive" if quality_score >= 0.6 else "negative",
                detail=f"Average quality score: {quality_score:.3f}",
            ),
            ValuationFactor(
                name="avg_compliance",
                raw_score=compliance_score,
                weight=self.COMPLIANCE_WEIGHT,
                weighted_score=compliance_score * self.COMPLIANCE_WEIGHT,
                impact="positive" if compliance_score >= 0.8 else "neutral",
                detail=f"Average compliance score: {compliance_score:.3f}",
            ),
        ]

        composite = sum(f.weighted_score for f in factors)
        base_value = self.BASE_UNIT_VALUE * n * composite

        desen_penalty = 0.0
        compliance_bonus = 0.0
        gov_explanation = ""
        rp = product.rights_summary
        if rp:
            if rp.desensitization_required:
                desen_penalty = base_value * 0.15
                gov_explanation += "15% desensitization penalty applied. "
            if product.avg_compliance >= 0.9:
                compliance_bonus = base_value * 0.10
                gov_explanation += "10% compliance bonus applied. "

        final = base_value - desen_penalty + compliance_bonus

        gov_impact = GovernanceImpactOnValue(
            tradable=product.tradable,
            desensitization_penalty=round(desen_penalty, 2),
            compliance_bonus=round(compliance_bonus, 2),
            explanation=gov_explanation.strip() or "No governance adjustments.",
        )

        readiness = "ready" if product.tradable and final > 0 else "not_ready"

        return ValuationExplanation(
            asset_unit_id=product.product_id,
            factors=factors,
            governance_impact=gov_impact,
            quality_overall=round(quality_score, 4),
            value_overall=round(composite, 4),
            final_value=round(final, 2),
            asset_class_rationale=f"Product category: {product.product_category.value}",
            promotion_readiness=readiness,
        )

"""Structured valuation explanation models.

These complement the numeric ValuationResultV2 with human-readable,
machine-queryable rationale for each dimension of the score.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ValuationFactor(BaseModel):
    """A single scored dimension contributing to the valuation."""
    name: str
    raw_score: float = 0.0
    weight: float = 0.0
    weighted_score: float = 0.0
    impact: str = "neutral"  # positive | negative | neutral
    detail: str = ""


class GovernanceImpactOnValue(BaseModel):
    """How governance constraints affect the estimated value."""
    tradable: bool = False
    desensitization_penalty: float = 0.0
    compliance_bonus: float = 0.0
    explanation: str = ""


class ValuationExplanation(BaseModel):
    """Full explanation for a valuation result."""
    asset_unit_id: str
    factors: List[ValuationFactor] = Field(default_factory=list)
    governance_impact: Optional[GovernanceImpactOnValue] = None
    quality_overall: float = 0.0
    value_overall: float = 0.0
    final_value: float = 0.0
    asset_class_rationale: str = ""
    promotion_readiness: str = ""

    def summary(self) -> str:
        lines = [f"Valuation explanation for {self.asset_unit_id}:"]
        lines.append(f"  Quality: {self.quality_overall:.2f}  Value: {self.value_overall:.2f}")
        lines.append(f"  Final estimated value: {self.final_value:.2f}")
        if self.governance_impact:
            lines.append(f"  Governance: {self.governance_impact.explanation}")
        for f in self.factors:
            lines.append(f"  [{f.impact:>8}] {f.name}: {f.raw_score:.2f} * {f.weight:.2f} = {f.weighted_score:.2f} — {f.detail}")
        return "\n".join(lines)

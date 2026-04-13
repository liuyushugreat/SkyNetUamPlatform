"""Multi-dimensional valuation result models.

Valuation is a two-phase process:

1. **Data quality scoring** — measures how *good* the underlying data is
   (completeness, sensor reliability, compliance …).
2. **Asset value scoring** — measures how *useful / rare / reusable* the
   data is in specific business scenarios.

Both scores feed into a final :class:`ValuationResultV2` which carries the
estimated monetary value together with a confidence indicator.

.. note::

   The class is named ``ValuationResultV2`` to avoid shadowing the legacy
   ``ValuationResult`` from ``SkyRwa.valuation`` (the original module-root
   file) which is still re-exported for backward compatibility.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class DataQualityScore(BaseModel):
    """Intrinsic quality dimensions of flight data."""
    completeness: float = 0.0
    temporal_continuity: float = 0.0
    sensor_reliability: float = 0.0
    event_richness: float = 0.0
    compliance_degree: float = 0.0
    overall: float = 0.0


class AssetValueScore(BaseModel):
    """Extrinsic / market-facing value dimensions."""
    scarcity: float = 0.0
    scenario_relevance: float = 0.0
    reuse_potential: float = 0.0
    timeliness: float = 0.0
    overall: float = 0.0


class ValuationResultV2(BaseModel):
    """Full valuation output for one :class:`FlightAssetUnit`."""
    asset_unit_id: str
    quality_score: DataQualityScore = Field(default_factory=DataQualityScore)
    value_score: AssetValueScore = Field(default_factory=AssetValueScore)
    estimated_value: float = 0.0
    currency: str = "USD"
    confidence: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    engine_id: str = "rule_based"
    notes: str = ""

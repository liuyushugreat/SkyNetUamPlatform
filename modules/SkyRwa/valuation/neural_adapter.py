"""Adapter that wraps the legacy neural-pricing models for the new pipeline.

This module bridges the original ``neural_pricing.py`` (PyTorch-based pricing
models) into the :class:`AbstractAssetValuationEngine` contract so that
existing trained models can be used as one *signal* feeding into asset
valuation — without replacing the full rule-based scoring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetStatus
from ..models.valuation import AssetValueScore, DataQualityScore, ValuationResultV2
from .base import AbstractAssetValuationEngine
from .rule_engine import RuleBasedValuationEngine

if TYPE_CHECKING:
    import torch


class NeuralValuationAdapter(AbstractAssetValuationEngine):
    """
    Uses a legacy :class:`NeuralPricingModel` to produce a price signal, then
    merges it with the rule-based quality / value scores.

    Parameters
    ----------
    neural_model:
        Any ``nn.Module`` with a ``forward(time_idx, route_idx) -> Tensor``
        signature (e.g. ``PizzaPricingModel`` or ``TorusPricingModel``).
    time_mod:
        Modulus for the time index (must match the model's embedding size).
    route_mod:
        Modulus for the route index.
    neural_weight:
        Blending weight for the neural price vs. the rule-based estimate.
    """

    engine_id: str = "neural_adapter"

    def __init__(
        self,
        neural_model: "torch.nn.Module",
        time_mod: int = 24,
        route_mod: int = 60,
        neural_weight: float = 0.4,
        base_price: float = 100.0,
        currency: str = "USD",
    ):
        self.neural_model = neural_model
        self.time_mod = time_mod
        self.route_mod = route_mod
        self.neural_weight = neural_weight
        self._rule_engine = RuleBasedValuationEngine(
            base_price=base_price,
            currency=currency,
        )
        self.base_price = base_price
        self.currency = currency

    def evaluate(self, unit: FlightAssetUnit) -> ValuationResultV2:
        import torch

        rule_result = self._rule_engine.evaluate(unit)

        time_idx = self._extract_time_idx(unit)
        route_idx = self._extract_route_idx(unit)

        t_tensor = torch.tensor([time_idx], dtype=torch.long)
        r_tensor = torch.tensor([route_idx], dtype=torch.long)

        self.neural_model.eval()
        with torch.no_grad():
            neural_price = float(self.neural_model(t_tensor, r_tensor).squeeze())

        neural_price = max(neural_price, 0.0)

        blended = (
            (1 - self.neural_weight) * rule_result.estimated_value
            + self.neural_weight * neural_price
        )

        result = ValuationResultV2(
            asset_unit_id=unit.asset_unit_id,
            quality_score=rule_result.quality_score,
            value_score=rule_result.value_score,
            estimated_value=round(blended, 4),
            currency=self.currency,
            confidence=round(rule_result.confidence * 0.9, 4),
            breakdown={
                **rule_result.breakdown,
                "neural_raw_price": neural_price,
                "neural_weight": self.neural_weight,
                "blended_value": blended,
            },
            engine_id=self.engine_id,
            notes="Blended rule-based + neural pricing",
        )

        unit.valuation_result = result
        unit.data_quality_score = rule_result.quality_score.overall
        unit.status = AssetStatus.VALUATED
        unit.updated_at = datetime.now(UTC)
        return result

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_time_idx(self, unit: FlightAssetUnit) -> int:
        if unit.start_time:
            return unit.start_time.hour % self.time_mod
        return 0

    def _extract_route_idx(self, unit: FlightAssetUnit) -> int:
        raw = hash(unit.flight_id) % self.route_mod
        return raw

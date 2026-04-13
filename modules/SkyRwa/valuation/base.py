"""Abstract base for asset-level valuation engines.

This replaces the legacy :class:`AbstractValuationEngine` (which operated on
``DataPacket`` objects) with a richer contract that works on
:class:`FlightAssetUnit` and produces :class:`ValuationResultV2`.

The old interface is **not** removed — it is still importable from
``SkyRwa.valuation_legacy`` (the original root-level ``valuation.py``) for
backward compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.asset_unit import FlightAssetUnit
from ..models.valuation import ValuationResultV2


class AbstractAssetValuationEngine(ABC):
    """Contract for flight-asset valuation engines."""

    engine_id: str = "abstract"

    @abstractmethod
    def evaluate(self, unit: FlightAssetUnit) -> ValuationResultV2:
        """Produce a multi-dimensional valuation for *unit*."""
        ...

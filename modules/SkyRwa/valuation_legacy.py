"""
Legacy valuation interfaces (Phase-1 compatibility layer).

.. deprecated::
    These classes are superseded by the V2 pipeline:

    ============================================  ==================================================
    Legacy class                                  Migration target
    ============================================  ==================================================
    ``DataPacket``                                ``FlightIngestRecord`` + ``FlightEvidencePackage``
    ``ValuationResult``                           ``ValuationResultV2`` (in ``models.valuation``)
    ``AbstractValuationEngine.evaluate(packet)``  ``AbstractAssetValuationEngine.evaluate(unit)``
    ============================================  ==================================================

    Import path unchanged: ``from SkyRwa import DataPacket, ValuationResult``.

TODO(cleanup): remove this file once all downstream consumers have migrated
to the V2 models and ``AbstractAssetValuationEngine``.
"""

from dataclasses import dataclass
from typing import Dict, Any
from abc import ABC, abstractmethod


@dataclass
class DataPacket:
    """Legacy data-packet model.  Prefer ``FlightIngestRecord`` for new code."""
    id: str
    source: str
    destination: str
    size_bytes: int
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class ValuationResult:
    """Legacy valuation result.  Prefer ``ValuationResultV2`` for new code."""
    packet_id: str
    value: float
    currency: str
    confidence_score: float
    breakdown: Dict[str, float]


class AbstractValuationEngine(ABC):
    """Legacy abstract valuation engine.

    Prefer ``AbstractAssetValuationEngine`` (in ``SkyRwa.valuation.base``)
    which operates on ``FlightAssetUnit`` and returns ``ValuationResultV2``.
    """

    @abstractmethod
    def evaluate(self, packet: DataPacket) -> ValuationResult:
        """Calculate the value of a given data packet."""
        pass

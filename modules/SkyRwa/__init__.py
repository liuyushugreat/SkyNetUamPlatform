"""
SkyRwa Module
=============

This package is the stable home for SkyRwa / RWA finance primitives such as:
- Data packet valuation interfaces
- Dynamic pricing engines
- Airspace voxel congestion pricing models

Phase-1 migration keeps existing behavior intact by providing compatibility
re-exports from the old `nexus_core` paths.
"""

from .valuation import AbstractValuationEngine, DataPacket, ValuationResult
from .pricing_engine import PricingEngine
from .economics.pricing import CongestionPricingModel, VoxelParams
from .neural_pricing import PizzaPricingModel, TorusPricingModel, CyclicEmbedding
from .adversarial import ArbitrageInjector

try:
    from .topology_metrics import calculate_integrity_score, get_betti_numbers
except ImportError:  # pragma: no cover
    # Keep the package importable in lightweight environments where optional
    # topology-analysis dependencies such as `ripser` are unavailable.
    calculate_integrity_score = None  # type: ignore[assignment]
    get_betti_numbers = None  # type: ignore[assignment]

__all__ = [
    "AbstractValuationEngine",
    "DataPacket",
    "ValuationResult",
    "PricingEngine",
    "CongestionPricingModel",
    "VoxelParams",
    "PizzaPricingModel",
    "TorusPricingModel",
    "CyclicEmbedding",
    "ArbitrageInjector",
    "calculate_integrity_score",
    "get_betti_numbers",
]

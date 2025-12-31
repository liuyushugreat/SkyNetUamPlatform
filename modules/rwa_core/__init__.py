"""
RWA Core Module (Phase-1)
========================

This package is the stable home for RWA/finance primitives such as:
- Data packet valuation interfaces
- Dynamic pricing engines
- Airspace voxel congestion pricing models

Phase-1 migration keeps existing behavior intact by providing compatibility
re-exports from the old `nexus_core` paths.
"""

from .valuation import AbstractValuationEngine, DataPacket, ValuationResult
from .pricing_engine import PricingEngine
from .economics.pricing import CongestionPricingModel, VoxelParams

__all__ = [
    "AbstractValuationEngine",
    "DataPacket",
    "ValuationResult",
    "PricingEngine",
    "CongestionPricingModel",
    "VoxelParams",
]


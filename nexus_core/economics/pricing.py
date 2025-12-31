"""
Compatibility shim (Phase-1)
============================

This module used to host the congestion pricing model. It now re-exports the
implementation from `modules.rwa_core` to keep existing imports working.
"""

from modules.rwa_core.economics.pricing import CongestionPricingModel, VoxelParams  # noqa: F401

__all__ = ["CongestionPricingModel", "VoxelParams"]



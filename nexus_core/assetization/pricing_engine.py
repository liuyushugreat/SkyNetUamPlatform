"""
Compatibility shim (Phase-1)
============================

This module used to host the data pricing engine. It now re-exports the
implementation from `modules.rwa_core` to keep existing imports working.
"""

from modules.rwa_core.pricing_engine import PricingEngine  # noqa: F401

__all__ = ["PricingEngine"]



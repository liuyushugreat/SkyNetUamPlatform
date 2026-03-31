"""
Compatibility shim (Phase-1)
============================

This module used to host the data pricing engine. It now re-exports the
implementation from `modules.SkyRwa` to keep existing imports working.
"""

from modules.SkyRwa.pricing_engine import PricingEngine  # noqa: F401

__all__ = ["PricingEngine"]



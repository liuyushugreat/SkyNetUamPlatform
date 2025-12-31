"""
Compatibility shim (Phase-1)
============================

This module used to host the valuation interfaces. It now re-exports the
implementation from `modules.rwa_core` to keep existing imports working.
"""

# Keep TelemetryEvent available for any legacy code that imported it from here.
from .event_bus import TelemetryEvent  # noqa: F401

from modules.rwa_core.valuation import (  # noqa: F401
    AbstractValuationEngine,
    DataPacket,
    ValuationResult,
)

__all__ = [
    "TelemetryEvent",
    "AbstractValuationEngine",
    "DataPacket",
    "ValuationResult",
]



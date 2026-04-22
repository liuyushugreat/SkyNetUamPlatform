"""SkyGrid — Spatio-Temporally Partitioned Edge-Cloud Runtime for
Hybrid Neural-Symbolic Reasoning Pipelines.

Three mechanisms (STP, COP, ABP) are exposed through a unified runtime
(`SkyGridRuntime`) and a discrete-event simulator (`Fabric`).  The public
API below is the only surface tests and scripts should import from.
"""

from __future__ import annotations

from .config import SkyGridConfig
from .runtime.engine import SkyGridRuntime
from .simulator.fabric import Fabric
from .telemetry.metrics import RunMetrics

__all__ = [
    "SkyGridConfig",
    "SkyGridRuntime",
    "Fabric",
    "RunMetrics",
]

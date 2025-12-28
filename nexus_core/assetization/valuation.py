from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
from .event_bus import TelemetryEvent

@dataclass
class DataPacket:
    """Represents a chunk of valid flight data ready for valuation."""
    source_did: str
    data_type: str        # "TRAJECTORY", "VIDEO", "ENVIRONMENT_SENSOR"
    quality_score: float  # 0.0 to 1.0 (based on sensor health/noise)
    payload: Dict[str, Any]
    timestamp: float

@dataclass
class ValuationResult:
    """Output of the valuation engine."""
    asset_id: str
    estimated_price: float
    currency: str = "SKY"
    metadata: Dict[str, Any] = None

class AbstractValuationEngine(ABC):
    """
    Interface for dynamic data pricing.
    """

    @abstractmethod
    def evaluate(self, packet: DataPacket) -> ValuationResult:
        """
        Calculates the monetary value of a data packet based on market demand,
        quality, and scarcity.
        """
        pass

    @abstractmethod
    def update_market_params(self, params: Dict[str, float]):
        """
        Updates internal pricing models (e.g., base rates, demand multipliers).
        """
        pass


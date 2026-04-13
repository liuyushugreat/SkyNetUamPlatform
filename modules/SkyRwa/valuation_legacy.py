from dataclasses import dataclass
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

@dataclass
class DataPacket:
    id: str
    source: str
    destination: str
    size_bytes: int
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class ValuationResult:
    packet_id: str
    value: float
    currency: str
    confidence_score: float
    breakdown: Dict[str, float]

class AbstractValuationEngine(ABC):
    """
    Abstract base class for Valuation Engines.
    Determines the monetary value of data packets or assets.
    """
    
    @abstractmethod
    def evaluate(self, packet: DataPacket) -> ValuationResult:
        """
        Calculate the value of a given data packet.
        """
        pass


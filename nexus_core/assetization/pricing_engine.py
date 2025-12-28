import uuid
import time
from typing import Dict, Any
from .valuation import AbstractValuationEngine, DataPacket, ValuationResult

class PricingEngine(AbstractValuationEngine):
    """
    Dynamic Pricing Engine for SkyNet Data Assets.
    Formula: P = Base * Quality * (1 + Scarcity) * (1 / (1 + Latency)) * DemandMultiplier
    """

    def __init__(self):
        # Default market parameters
        self.market_params = {
            "base_rate": 1.0,           # Base price per packet
            "scarcity_multiplier": 2.0, # Multiplier for rare data (e.g., downtown)
            "demand_factor": 1.0        # Global demand index
        }
        
        # Define high-value zones (Simple bounding boxes for simulation)
        # Format: (min_x, max_x, min_y, max_y)
        self.high_value_zones = [
            (40, 60, 40, 60), # City Center (High Scarcity)
        ]

    def evaluate(self, packet: DataPacket) -> ValuationResult:
        """
        Calculates price based on latency, scarcity, and quality.
        """
        # 1. Calculate Latency Decay
        current_time = time.time()
        # Ensure non-negative latency (clock skew protection)
        latency = max(0.0, current_time - packet.timestamp)
        # Decay factor: Price drops as data gets older. 
        # Example: 1s delay = 0.5x value, 0s delay = 1.0x value
        latency_factor = 1.0 / (1.0 + latency)

        # 2. Calculate Spatial Scarcity
        scarcity_score = 0.0
        pos = packet.payload.get("pos", [0, 0, 0])
        if len(pos) >= 2:
            x, y = pos[0], pos[1]
            for zone in self.high_value_zones:
                if zone[0] <= x <= zone[1] and zone[2] <= y <= zone[3]:
                    scarcity_score = 1.0 # Hit a high-value zone
                    break
        
        # 3. Calculate Final Price
        # P = Base * Quality * (1 + Scarcity * Multiplier) * LatencyFactor * Demand
        base = self.market_params["base_rate"]
        quality = packet.quality_score
        scarcity_mult = self.market_params["scarcity_multiplier"]
        demand = self.market_params["demand_factor"]

        price = base * quality * (1 + scarcity_score * scarcity_mult) * latency_factor * demand

        # 4. Generate Metadata
        metadata = {
            "latency_ms": round(latency * 1000, 2),
            "scarcity_level": "HIGH" if scarcity_score > 0 else "NORMAL",
            "quality_score": quality,
            "data_type": packet.data_type,
            "generation_time": packet.timestamp,
            "pricing_model_version": "v1.0"
        }

        return ValuationResult(
            asset_id=f"asset-{uuid.uuid4().hex[:8]}",
            estimated_price=round(price, 4),
            currency="SKY",
            metadata=metadata
        )

    def update_market_params(self, params: Dict[str, float]):
        """
        Updates market parameters dynamically.
        """
        self.market_params.update(params)


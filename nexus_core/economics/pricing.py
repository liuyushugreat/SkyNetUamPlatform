"""
SkyNet-RWA-Nexus Economics Engine: Dynamic Pricing Models.

This module implements the congestion pricing algorithms based on 
supply-demand curves in 3D voxel space.

Mathematical Model (Voxel Congestion Pricing):
    The price $P(v, t)$ for traversing voxel $v$ at time $t$ is defined as:
    
    $$ P(v, t) = P_{base} + \alpha \cdot e^{\beta \frac{N(v, t)}{C(v)}} $$
    
    Where:
    - $P_{base}$: Base toll fee for the airspace class.
    - $N(v, t)$: Current number of active agents in voxel $v$.
    - $C(v)$: Safety capacity limit of voxel $v$ (saturation point).
    - $\alpha$: Sensitivity coefficient (Elasticity scaling factor).
    - $\beta$: Exponential penalty factor for high congestion.
"""

import math
from typing import Dict
from dataclasses import dataclass

@dataclass
class VoxelParams:
    voxel_id: str
    base_price: float
    capacity: int
    sensitivity_alpha: float = 2.0
    penalty_beta: float = 3.5

class CongestionPricingModel:
    """
    Implements dynamic pricing logic for Airspace Voxels.
    """

    def __init__(self):
        self._voxel_registry: Dict[str, VoxelParams] = {}

    def register_voxel(self, params: VoxelParams) -> None:
        """Initialize a voxel with its economic parameters."""
        self._voxel_registry[params.voxel_id] = params

    def calculate_toll(self, voxel_id: str, current_occupancy: int) -> float:
        """
        Computes the real-time toll fee $P(v, t)$.

        Args:
            voxel_id: The identifier of the 3D grid cell.
            current_occupancy: $N(v, t)$, current count of UAVs.

        Returns:
            float: Calculated price in SkyToken units (wei).
        
        Raises:
            ValueError: If voxel is not registered.
        """
        if voxel_id not in self._voxel_registry:
            raise ValueError(f"Voxel {voxel_id} not registered in Economic Engine.")

        params = self._voxel_registry[voxel_id]
        
        # Avoid division by zero
        capacity = max(1, params.capacity)
        
        # Calculate utilization ratio $\rho = N / C$
        utilization_ratio = current_occupancy / capacity
        
        # Apply Formula: P_base + alpha * exp(beta * ratio)
        congestion_premium = params.sensitivity_alpha * math.exp(params.penalty_beta * utilization_ratio)
        
        total_price = params.base_price + congestion_premium
        
        return round(total_price, 4)

    def get_marginal_cost(self, voxel_id: str, current_occupancy: int) -> float:
        """
        Calculates the marginal cost of adding one more agent.
        $$ MC = P(n+1) - P(n) $$
        """
        current_price = self.calculate_toll(voxel_id, current_occupancy)
        next_price = self.calculate_toll(voxel_id, current_occupancy + 1)
        return next_price - current_price


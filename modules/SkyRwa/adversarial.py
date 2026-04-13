"""
Adversarial / synthetic-anomaly injectors (Phase-1 — retained).

These utilities generate synthetic market anomalies for testing the
topological integrity of pricing manifolds.  They are used in conjunction
with ``topology_metrics.py`` and are not part of the V2 asset pipeline.

FIXME(scope): consider moving to a ``testing/`` or ``experiments/``
sub-package once the V2 pipeline stabilises.
"""

import numpy as np
import torch
from typing import Tuple, Optional

class ArbitrageInjector:
    """
    Injects synthetic market anomalies to simulate adversarial conditions 
    or structural failures in the pricing manifold.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def inject_cyclic_loop(
        self, 
        time_data: torch.Tensor, 
        route_data: torch.Tensor, 
        price_data: torch.Tensor,
        num_victims: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates a 'Money Pump' arbitrage loop.
        
        Logic:
        Selects a sequence of points and forces a price gradient that violates 
        conservative vector field properties locally, creating a localized 
        non-conservative loop (vortex) in the manifold.
        
        Effect on Topology:
        May introduce an extra H1 generator (cycle) if the loop is large enough,
        or simply distort the geometry.
        """
        t_mod = time_data.clone()
        r_mod = route_data.clone()
        p_mod = price_data.clone()
        
        # Select random indices to form a loop
        indices = self.rng.choice(len(time_data), num_victims, replace=False)
        
        # Force a local violation: Price should roughly depend on T+R.
        # We invert this relationship for the selected points to create a "hole" or "twist"
        # For example, setting Price = - (Time + Route) locally while globally it's positive.
        
        # A clearer arbitrage: A -> B -> C -> A where profit > 0.
        # In a pricing surface context, this looks like a "Escher Staircase" locally.
        # We simulate this by adding a large step function modulo the loop size.
        
        distortion = torch.linspace(0, 10, steps=num_victims)
        # Apply distortion to the selected indices
        p_mod[indices] = p_mod[indices] + distortion.view(-1, 1)
        
        # To close the loop (make it nasty), we force the last few points 
        # to drop significantly below the first points.
        p_mod[indices[-10:]] = p_mod[indices[0]] - 5.0
        
        return t_mod, r_mod, p_mod

    def inject_fragmentation(
        self, 
        time_data: torch.Tensor, 
        route_data: torch.Tensor, 
        price_data: torch.Tensor,
        gap_size: float = 5.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates a liquidity fracture (fragmentation).
        
        Logic:
        Identifies a specific region in the Time-Route grid and shifts its prices
        dramatically, creating a disconnected component in the value manifold.
        
        Effect on Topology:
        Increases Beta_0 (Connected Components) > 1.
        """
        t_mod = time_data.clone()
        r_mod = route_data.clone()
        p_mod = price_data.clone()
        
        # Define a region to fracture (e.g., specific time window)
        # e.g., Time slots 10-14
        mask = (t_mod >= 10) & (t_mod <= 14)
        
        # Shift prices up significantly to create a gap
        p_mod[mask] += gap_size
        
        return t_mod, r_mod, p_mod


"""Sinusoidal temporal encoding φ(δ) for elapsed-time edge features.

Analogous to positional encoding in Transformers (Vaswani et al., 2017)
but applied to elapsed time since edge last observed, enabling TR-GAT
to discount stale relational edges without explicit graph surgery.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalTemporalEncoding(nn.Module):
    """Maps scalar temporal offset δ to a d_phi-dimensional embedding."""

    def __init__(self, d_phi: int = 32, max_period: float = 300.0):
        super().__init__()
        self.d_phi = max(d_phi, 2)
        half = self.d_phi // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * -(math.log(max_period) / half)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta: Tensor of shape (...,) – elapsed time in seconds.
        Returns:
            Tensor of shape (..., d_phi).
        """
        delta = delta.unsqueeze(-1)
        args = delta * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

"""Sinusoidal temporal encoding φ(δ) — Equation (2) in the paper.

Implements:
  φ(δ)_{2k}   = sin(δ / T^{2k/d_φ})
  φ(δ)_{2k+1} = cos(δ / T^{2k/d_φ})

where δ is the elapsed time (seconds) since an edge was last observed,
d_φ = 32 is the encoding dimension, and T = 300 s is the maximum period.
This allows TR-GAT to smoothly discount stale relational edges without
hard graph surgery, analogous to positional encoding in Transformers.

Reference: Section 4.1, Equation (2) in the paper.
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

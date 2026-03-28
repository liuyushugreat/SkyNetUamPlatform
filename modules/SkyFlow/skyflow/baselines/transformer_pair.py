"""Transformer-Pair baseline.

Replaces the LSTM with a Transformer encoder using the same pairwise
scoring strategy. Enables parallelizable long-range dependency modeling
but treats each aircraft independently without relational structure.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from skyflow.data.tkg_builder import TKGSnapshot


class TransformerPair(nn.Module):
    """Transformer-based pairwise conflict detector."""

    def __init__(
        self,
        input_dim: int = 23,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        snapshot: TKGSnapshot,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_uav = snapshot.num_uavs
        feats = snapshot.node_features[:n_uav]

        if history is None:
            history = feats.unsqueeze(1)

        x = self.input_proj(history)
        encoded = self.encoder(x)
        embeddings = encoded[:, -1, :]

        pairs = snapshot.conflict_pairs
        if pairs is None or pairs.size(1) == 0:
            return torch.zeros(0, device=feats.device)

        h_i = embeddings[pairs[0]]
        h_j = embeddings[pairs[1]]
        x = torch.cat([h_i, h_j], dim=-1)
        return torch.sigmoid(self.scorer(x)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""LSTM-Pair baseline.

Independent LSTM encoders on each aircraft's position-velocity history.
Conflict scores from concatenated final hidden states. Captures temporal
dynamics for individual trajectories but has no relational structure.
Parameter count matched to TR-GAT (~4.2M) for fair comparison.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from skyflow.data.tkg_builder import TKGSnapshot


class LSTMPair(nn.Module):
    """LSTM-based pairwise conflict detector."""

    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
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
        """
        Args:
            snapshot: Current TKG snapshot.
            history: (N_uav, T, D) trajectory history or None (uses current).
        Returns:
            (P,) conflict probabilities.
        """
        n_uav = snapshot.num_uavs
        feats = snapshot.node_features[:n_uav]

        if history is None:
            history = feats.unsqueeze(1)

        _, (h_n, _) = self.encoder(history)
        embeddings = h_n[-1]

        pairs = snapshot.conflict_pairs
        if pairs is None or pairs.size(1) == 0:
            return torch.zeros(0, device=feats.device)

        h_i = embeddings[pairs[0]]
        h_j = embeddings[pairs[1]]
        x = torch.cat([h_i, h_j], dim=-1)
        return torch.sigmoid(self.scorer(x)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

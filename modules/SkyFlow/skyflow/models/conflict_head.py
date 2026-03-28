"""Pairwise conflict scoring head.

Takes final-layer TR-GAT embeddings, recurrent state, and optional
direct edge features to produce conflict probability for each UAV pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConflictScoringHead(nn.Module):
    """2-layer MLP producing per-pair conflict probability.

    Input for pair (i,j):
        [h_i ‖ h_j ‖ s_i ‖ s_j ‖ e_ij]
    where h are final-layer embeddings, s are recurrent states,
    and e_ij is the direct proximity edge feature (or learned no-edge token).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        recurrent_dim: int = 64,
        edge_feature_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = 2 * embed_dim + 2 * recurrent_dim + edge_feature_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.no_edge_token = nn.Parameter(torch.randn(edge_feature_dim) * 0.01)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        s_i: torch.Tensor,
        s_j: torch.Tensor,
        edge_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h_i, h_j: (P, embed_dim) embeddings for source/target UAVs.
            s_i, s_j: (P, recurrent_dim) recurrent states.
            edge_feat: (P, edge_feature_dim) or None → uses no-edge token.
        Returns:
            (P,) conflict probabilities in [0, 1].
        """
        P = h_i.size(0)
        if edge_feat is None:
            edge_feat = self.no_edge_token.unsqueeze(0).expand(P, -1)
        x = torch.cat([h_i, h_j, s_i, s_j, edge_feat], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)

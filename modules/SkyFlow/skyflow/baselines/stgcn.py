"""Spatio-Temporal Graph Convolutional Network (STGCN) baseline.

Fixed 2D adjacency over proximity pairs without typed relations
or temporal edge weighting. Validates the benefit of graph structure
but lacks the relational and temporal sophistication of TR-GAT.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyflow.data.tkg_builder import TKGSnapshot


class GraphConvLayer(nn.Module):
    """Simple GCN layer: h' = σ(AXW)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, adj_src: torch.Tensor, adj_dst: torch.Tensor
    ) -> torch.Tensor:
        N = x.size(0)
        h = self.weight(x)

        if adj_src.size(0) > 0:
            agg = torch.zeros_like(h)
            messages = h[adj_src]
            idx = adj_dst.unsqueeze(-1).expand_as(messages)
            agg.scatter_add_(0, idx, messages)

            degree = torch.zeros(N, 1, device=x.device)
            degree.scatter_add_(0, adj_dst.unsqueeze(-1), torch.ones(adj_src.size(0), 1, device=x.device))
            degree = degree.clamp(min=1)

            h = h + agg / degree

        return self.dropout(F.relu(h + self.bias))


class STGCN(nn.Module):
    """Spatio-temporal GCN with fixed proximity adjacency."""

    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 128,
        num_gcn_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim, dropout))

        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, snapshot: TKGSnapshot) -> torch.Tensor:
        n_uav = snapshot.num_uavs
        x = self.input_proj(snapshot.node_features[:n_uav])

        adj_src, adj_dst = self._merge_edges(snapshot.edge_indices, n_uav)

        for gcn in self.gcn_layers:
            x = gcn(x, adj_src, adj_dst)

        x = self.temporal_conv(x.unsqueeze(-1)).squeeze(-1)

        pairs = snapshot.conflict_pairs
        if pairs is None or pairs.size(1) == 0:
            return torch.zeros(0, device=x.device)

        h_i = x[pairs[0]]
        h_j = x[pairs[1]]
        inp = torch.cat([h_i, h_j], dim=-1)
        return torch.sigmoid(self.scorer(inp)).squeeze(-1)

    def _merge_edges(
        self, edge_indices: Dict[int, torch.Tensor], n_uav: int
    ) -> tuple:
        all_src, all_dst = [], []
        for r, edges in edge_indices.items():
            src, dst = edges[0], edges[1]
            mask = (src < n_uav) & (dst < n_uav)
            all_src.append(src[mask])
            all_dst.append(dst[mask])

        if all_src:
            return torch.cat(all_src), torch.cat(all_dst)
        device = next(iter(edge_indices.values())).device if edge_indices else torch.device("cpu")
        return torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, dtype=torch.long, device=device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

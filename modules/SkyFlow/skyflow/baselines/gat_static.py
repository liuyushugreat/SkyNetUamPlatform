"""GAT-Static baseline.

Standard graph attention network on current-epoch snapshot without
temporal encoding or relation-type separation. Validates the benefit
of attention over fixed adjacency but misses temporal dynamics.
Also serves as the base for the TR-GAT-NoTemp ablation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyflow.data.tkg_builder import TKGSnapshot


class GATLayer(nn.Module):
    """Standard GAT layer (Velickovic et al., 2018) without temporal encoding."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim) * 0.01)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.head_dim)

        if src.size(0) == 0:
            return self.layer_norm(self.residual_proj(x))

        h_src = h[src]
        h_dst = h[dst]
        attn_input = torch.cat([h_src, h_dst], dim=-1)
        e = (attn_input * self.attn.unsqueeze(0)).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=0.2)

        e_max = torch.full((N, self.num_heads), -1e9, device=x.device)
        idx = dst.unsqueeze(-1).expand_as(e)
        e_max.scatter_reduce_(0, idx, e, reduce="amax", include_self=False)
        e_shifted = e - e_max.gather(0, idx)
        exp_e = torch.exp(e_shifted)
        sum_exp = torch.zeros(N, self.num_heads, device=x.device)
        sum_exp.scatter_add_(0, idx, exp_e)
        alpha = exp_e / (sum_exp.gather(0, idx) + 1e-12)
        alpha = self.dropout(alpha)

        msg = alpha.unsqueeze(-1) * h[src]
        agg = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        dst_exp = dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg)
        agg.scatter_add_(0, dst_exp, msg)

        out = agg.reshape(N, -1)
        residual = self.residual_proj(x)
        return self.layer_norm(residual + self.dropout(out))


class GATStatic(nn.Module):
    """Multi-layer GAT for conflict detection (no temporal encoding)."""

    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

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
        x = self.input_proj(snapshot.node_features)

        src, dst = self._merge_edges(snapshot.edge_indices)

        for layer in self.layers:
            x = layer(x, src, dst)

        x_uav = x[:n_uav]

        pairs = snapshot.conflict_pairs
        if pairs is None or pairs.size(1) == 0:
            return torch.zeros(0, device=x.device)

        h_i = x_uav[pairs[0]]
        h_j = x_uav[pairs[1]]
        inp = torch.cat([h_i, h_j], dim=-1)
        return torch.sigmoid(self.scorer(inp)).squeeze(-1)

    def _merge_edges(self, edge_indices: Dict[int, torch.Tensor]) -> tuple:
        all_src, all_dst = [], []
        for r, edges in edge_indices.items():
            all_src.append(edges[0])
            all_dst.append(edges[1])
        if all_src:
            return torch.cat(all_src), torch.cat(all_dst)
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

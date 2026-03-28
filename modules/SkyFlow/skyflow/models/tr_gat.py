"""TR-GAT: Temporally-conditioned Relational Graph Attention Network.

Core architecture of SkyFlow (Section 4 / Algorithm 2 in the paper).
Each TR-GAT layer computes multi-head attention conditioned on both
relation type r and sinusoidal temporal encoding φ(δ):

  Attention (Eq. 3):
    α_{ij}^{r,m} = softmax_j( LeakyReLU( a_r^m · [W_Q^r h_i || W_K^r h_j || φ(δ_{ij})] ) )

  Message aggregation (Eq. 4):
    z_i^r = Σ_j α_{ij}^{r,m} · W_V^r h_j

  Multi-relation gating (Eq. 5):
    h_i' = LayerNorm( h_i + Σ_r g_r(h_i) · z_i^r )
    where g_r = softmax(W_gate · [z_i^1 || ... || z_i^R])

A GRUCell produces recurrent state s_i across K=10 observation epochs.

Reference: Section 4.1–4.3 and Algorithm 2 in the paper.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyflow.models.temporal_encoding import SinusoidalTemporalEncoding

RELATION_TYPES = [
    "approaches",
    "conflicts_with",
    "shares_corridor",
    "is_downwind_of",
    "has_reserved",
    "is_restricted_by",
]


class TRGATLayer(nn.Module):
    """Single TR-GAT layer implementing Equations (3)–(5).

    Per-relation attention (Eq. 3): for each relation r ∈ {1..R},
    computes temporally-conditioned attention coefficients using
    per-relation Q/K/V projections and the sinusoidal encoding φ(δ).
    Relation outputs are fused via a learned softmax gate (Eq. 5).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_relations: int = 6,
        temporal_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W_Q = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)
        ])
        self.W_K = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)
        ])
        self.W_V = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)
        ])

        attn_in_dim = 2 * self.head_dim + temporal_dim
        self.attn_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(num_heads, attn_in_dim) * 0.01)
            for _ in range(num_relations)
        ])

        self.gate_proj = nn.Linear(out_dim * num_relations, num_relations)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: Dict[int, torch.Tensor],
        temporal_enc: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node embeddings.
            edge_indices: {relation_idx: (2, E_r)} source→target index pairs.
            temporal_enc: {relation_idx: (E_r, d_phi)} encoded elapsed time per edge.
        Returns:
            (N, out_dim) updated node embeddings.
        """
        N = x.size(0)
        device = x.device
        out_dim = self.head_dim * self.num_heads

        relation_outputs = []
        for r in range(self.num_relations):
            if r not in edge_indices or edge_indices[r].size(1) == 0:
                relation_outputs.append(torch.zeros(N, out_dim, device=device))
                continue

            src, dst = edge_indices[r]
            E_r = src.size(0)

            q = self.W_Q[r](x[dst]).view(E_r, self.num_heads, self.head_dim)
            k = self.W_K[r](x[src]).view(E_r, self.num_heads, self.head_dim)
            v = self.W_V[r](x[src]).view(E_r, self.num_heads, self.head_dim)

            phi = temporal_enc[r]
            phi_expanded = phi.unsqueeze(1).expand(-1, self.num_heads, -1)

            attn_input = torch.cat([q, k, phi_expanded], dim=-1)
            attn_logits = (attn_input * self.attn_vectors[r].unsqueeze(0)).sum(dim=-1)
            attn_logits = F.leaky_relu(attn_logits, negative_slope=0.2)

            attn_scores = _scatter_softmax(attn_logits, dst, N)
            attn_scores = self.dropout(attn_scores)

            messages = attn_scores.unsqueeze(-1) * v
            agg = torch.zeros(N, self.num_heads, self.head_dim, device=device)
            dst_expanded = dst.unsqueeze(-1).unsqueeze(-1).expand_as(messages)
            agg.scatter_add_(0, dst_expanded, messages)

            relation_outputs.append(agg.reshape(N, out_dim))

        stacked = torch.cat(relation_outputs, dim=-1)
        gate_logits = self.gate_proj(stacked)
        gates = F.softmax(gate_logits, dim=-1)

        fused = torch.zeros(N, out_dim, device=device)
        for r in range(self.num_relations):
            fused = fused + gates[:, r].unsqueeze(-1) * relation_outputs[r]

        residual = self.residual_proj(x)
        return self.layer_norm(residual + self.dropout(fused))


class TRGAT(nn.Module):
    """Full TR-GAT model (Algorithm 2): L stacked layers + GRU temporal summary.

    Architecture: input_proj → L × TRGATLayer → GRUCell
    The GRU carries recurrent state s_i across K consecutive TKG snapshots
    within each observation window (K=10 by default, i.e., 1 second at 10 Hz).
    """

    def __init__(
        self,
        node_feature_dim: int = 23,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_relations: int = 6,
        temporal_dim: int = 32,
        recurrent_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.recurrent_dim = recurrent_dim

        self.input_proj = nn.Linear(node_feature_dim, embed_dim)

        self.layers = nn.ModuleList([
            TRGATLayer(
                in_dim=embed_dim,
                out_dim=embed_dim,
                num_heads=num_heads,
                num_relations=num_relations,
                temporal_dim=temporal_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.temporal_gru = nn.GRUCell(embed_dim, recurrent_dim)
        self.temporal_encoding = SinusoidalTemporalEncoding(d_phi=temporal_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: Dict[int, torch.Tensor],
        edge_deltas: Dict[int, torch.Tensor],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (N, D) raw features for all entities at current epoch.
            edge_indices: {relation_idx: (2, E_r)} edges.
            edge_deltas: {relation_idx: (E_r,)} elapsed seconds per edge.
            recurrent_state: (N, recurrent_dim) or None for first epoch.
        Returns:
            node_emb: (N, embed_dim) final-layer embeddings.
            new_state: (N, recurrent_dim) updated recurrent summary.
        """
        x = self.input_proj(node_features)

        temporal_enc = {
            r: self.temporal_encoding(edge_deltas[r])
            for r in edge_deltas
        }

        for layer in self.layers:
            x = layer(x, edge_indices, temporal_enc)

        N = x.size(0)
        if recurrent_state is None:
            recurrent_state = torch.zeros(N, self.recurrent_dim, device=x.device)
        new_state = self.temporal_gru(x, recurrent_state)

        return x, new_state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _scatter_softmax(
    logits: torch.Tensor, index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Numerically stable softmax over variable-size neighborhoods."""
    max_vals = torch.full((num_nodes, logits.size(1)), -1e9, device=logits.device)
    idx = index.unsqueeze(-1).expand_as(logits)
    max_vals.scatter_reduce_(0, idx, logits, reduce="amax", include_self=False)
    shifted = logits - max_vals.gather(0, idx)
    exp_vals = torch.exp(shifted)
    sum_exp = torch.zeros(num_nodes, logits.size(1), device=logits.device)
    sum_exp.scatter_add_(0, idx, exp_vals)
    return exp_vals / (sum_exp.gather(0, idx) + 1e-12)

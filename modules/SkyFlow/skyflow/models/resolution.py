"""Coordinated conflict resolution module — Algorithm 3 in the paper.

Given a conflict cluster K = {u_1, ..., u_M} (maximal connected subgraph
of UAV nodes with pairwise conflict scores > τ), generates avoidance
waypoint offsets Δp_i ∈ R^3 via Projected Gradient Descent (PGD):

  Objective (Section 4.3):
    L = λ₁ · L_sep + λ₂ · L_dev
    L_sep = Σ_{i<j} [ ReLU(d_h - ||p'_i - p'_j||_h) + ReLU(d_v - |z'_i - z'_j|) ]
    L_dev = (1/M) Σ_i ||Δp_i||²

  PGD update:
    Δp ← Proj_{[-δ_max, δ_max]}( Δp - η ∇_Δp L )

Parameters: λ₁=10, λ₂=1, δ_max=50 m, η=0.5, K_pgd=20 steps.
Warm-started from f_init(H) where H are TR-GAT embeddings.
Output: 12 bytes per UAV (3 × float32), fits in a single 5G packet.

Reference: Section 4.3 and Algorithm 3 in the paper.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResolutionModule(nn.Module):
    """PGD-based waypoint offset generator for conflict clusters (M ≤ 12).

    The init_net provides a warm-start from TR-GAT embeddings, reducing
    convergence from 80+ iterations (random init) to 20 (Section 4.3).
    """

    MAX_CLUSTER_SIZE = 12

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        max_offset_m: float = 50.0,
        pgd_steps: int = 20,
        pgd_lr: float = 0.5,
    ):
        super().__init__()
        self.max_offset_m = max_offset_m
        self.pgd_steps = pgd_steps
        self.pgd_lr = pgd_lr

        self.init_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        h_sep: float = 10.0,
        v_sep: float = 3.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Projected gradient descent per Algorithm 3 in the paper.

        Args:
            embeddings: (M, embed_dim) TR-GAT embeddings for cluster UAVs.
            positions: (M, 3) current [x, y, z] in meters.
            velocities: (M, 3) current velocity vectors.
            h_sep: horizontal separation minimum (meters).
            v_sep: vertical separation minimum (meters).
        Returns:
            offsets: (M, 3) waypoint offsets in meters.
            info: dict with optimization metadata.
        """
        M = embeddings.size(0)
        if M < 2:
            return torch.zeros(M, 3, device=embeddings.device), {"steps": 0}

        with torch.no_grad():
            init_offsets = self.init_net(embeddings)
            init_offsets = torch.clamp(init_offsets, -self.max_offset_m, self.max_offset_m)

        offsets = init_offsets.clone().requires_grad_(True)
        best_offsets = offsets.detach().clone()
        best_obj = float("inf")

        for step in range(self.pgd_steps):
            new_pos = positions + offsets
            loss = _resolution_objective(
                new_pos, positions, velocities, h_sep, v_sep
            )
            loss.backward()

            with torch.no_grad():
                offsets.data -= self.pgd_lr * offsets.grad
                offsets.data.clamp_(-self.max_offset_m, self.max_offset_m)
                offsets.grad.zero_()

                if loss.item() < best_obj:
                    best_obj = loss.item()
                    best_offsets = offsets.data.clone()

        return best_offsets, {"steps": self.pgd_steps, "final_loss": best_obj}


def _resolution_objective(
    new_pos: torch.Tensor,
    orig_pos: torch.Tensor,
    velocities: torch.Tensor,
    h_sep: float,
    v_sep: float,
) -> torch.Tensor:
    """Combined objective L = λ₁·L_sep + λ₂·L_dev (Algorithm 3, line 5)."""
    M = new_pos.size(0)

    path_dev = ((new_pos - orig_pos) ** 2).sum(dim=-1).mean()

    diffs = new_pos.unsqueeze(0) - new_pos.unsqueeze(1)
    h_dist = torch.sqrt(diffs[:, :, :2].pow(2).sum(dim=-1) + 1e-8)
    v_dist = torch.abs(diffs[:, :, 2])

    h_violation = F.relu(h_sep - h_dist)
    v_violation = F.relu(v_sep - v_dist)
    mask = 1.0 - torch.eye(M, device=new_pos.device)
    sep_penalty = ((h_violation + v_violation) * mask).sum()

    return sep_penalty * 10.0 + path_dev * 1.0

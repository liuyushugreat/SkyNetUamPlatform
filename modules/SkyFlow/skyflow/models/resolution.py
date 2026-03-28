"""Coordinated conflict resolution module.

Given a conflict cluster (maximal connected subgraph of UAVs with
pairwise conflict scores > τ), generates avoidance waypoint offsets
that maximize minimum separation while minimizing path deviation.
Uses projected gradient descent warm-started from TR-GAT embeddings.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResolutionModule(nn.Module):
    """Waypoint offset generator for conflict clusters up to size 12."""

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

        init_offsets = self.init_net(embeddings)
        init_offsets = torch.clamp(init_offsets, -self.max_offset_m, self.max_offset_m)
        offsets = init_offsets.detach().clone().requires_grad_(True)

        optimizer = torch.optim.Adam([offsets], lr=self.pgd_lr)

        best_offsets = offsets.detach().clone()
        best_obj = float("inf")

        for step in range(self.pgd_steps):
            optimizer.zero_grad()
            new_pos = positions + offsets

            loss = _resolution_objective(
                new_pos, positions, velocities, h_sep, v_sep
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                offsets.clamp_(-self.max_offset_m, self.max_offset_m)

            if loss.item() < best_obj:
                best_obj = loss.item()
                best_offsets = offsets.detach().clone()

        return best_offsets, {"steps": self.pgd_steps, "final_loss": best_obj}


def _resolution_objective(
    new_pos: torch.Tensor,
    orig_pos: torch.Tensor,
    velocities: torch.Tensor,
    h_sep: float,
    v_sep: float,
) -> torch.Tensor:
    """Combined objective: maximize min separation + minimize path deviation."""
    M = new_pos.size(0)

    path_dev = ((new_pos - orig_pos) ** 2).sum(dim=-1).mean()

    diffs = new_pos.unsqueeze(0) - new_pos.unsqueeze(1)
    h_dist = torch.sqrt(diffs[:, :, :2].pow(2).sum(dim=-1) + 1e-8)
    v_dist = torch.abs(diffs[:, :, 2])

    h_violation = F.relu(h_sep - h_dist)
    v_violation = F.relu(v_sep - v_dist)
    mask = 1.0 - torch.eye(M, device=new_pos.device)
    sep_penalty = ((h_violation + v_violation) * mask).sum()

    return sep_penalty * 10.0 + path_dev * 0.1

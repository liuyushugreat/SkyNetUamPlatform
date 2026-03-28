"""Focal loss for conflict detection — Section 5.3 in the paper.

Genuine conflicts constitute ~3.1% of all UAV pairs (Table 1).
Focal reweighting (Lin et al., 2017) with γ=2 and α=0.75
down-weights easy negatives and focuses model capacity on
ambiguous near-miss cases:

  FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

Reference: Section 5.3 in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) for imbalanced detection."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """
        Args:
            gamma: Focusing parameter — higher values more aggressively
                   down-weight easy examples.
            alpha: Balance factor for positive class weight.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (N,) predicted probabilities in [0, 1].
            target: (N,) ground-truth labels in {0, 1}.
        """
        pred = pred.clamp(1e-7, 1 - 1e-7)
        bce = F.binary_cross_entropy(pred, target, reduction="none")

        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        return (alpha_t * focal_weight * bce).mean()

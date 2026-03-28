"""Reciprocal Velocity Obstacle (VO) baseline.

Standard pairwise geometric method with 60-second look-ahead
and no semantic context. Fastest method (~8 ms) but lowest
detection quality due to pairwise-only reasoning.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from skyflow.data.tkg_builder import TKGSnapshot


class VelocityObstacle:
    """Deterministic reciprocal velocity obstacle conflict detector."""

    def __init__(
        self,
        lookahead: float = 60.0,
        h_sep: float = 10.0,
        v_sep: float = 3.0,
        threshold: float = 0.42,
    ):
        self.lookahead = lookahead
        self.h_sep = h_sep
        self.v_sep = v_sep
        self.threshold = threshold

    def predict(self, snapshot: TKGSnapshot) -> torch.Tensor:
        """Compute conflict scores for all evaluated pairs.

        Returns:
            (P,) conflict probability estimates in [0, 1].
        """
        feats = snapshot.node_features.cpu().numpy()
        n_uav = snapshot.num_uavs
        pairs = snapshot.conflict_pairs

        if pairs is None or pairs.size(1) == 0:
            return torch.zeros(0)

        positions = feats[:n_uav, 0:3]
        velocities = feats[:n_uav, 3:6]

        src = pairs[0].cpu().numpy()
        dst = pairs[1].cpu().numpy()
        scores = np.zeros(len(src), dtype=np.float32)

        for idx in range(len(src)):
            i, j = src[idx], dst[idx]
            if i >= n_uav or j >= n_uav:
                continue

            dp = positions[j] - positions[i]
            dv = velocities[j] - velocities[i]

            h_dist = np.sqrt(dp[0] ** 2 + dp[1] ** 2)
            v_dist = abs(dp[2])

            if np.dot(dv, dv) > 1e-8:
                t_cpa = -np.dot(dp, dv) / np.dot(dv, dv)
                t_cpa = np.clip(t_cpa, 0, self.lookahead)
                cpa = dp + dv * t_cpa
                cpa_h = np.sqrt(cpa[0] ** 2 + cpa[1] ** 2)
                cpa_v = abs(cpa[2])
            else:
                cpa_h = h_dist
                cpa_v = v_dist
                t_cpa = 0

            if cpa_h < self.h_sep * 2 and cpa_v < self.v_sep * 2:
                h_score = max(0, 1 - cpa_h / (self.h_sep * 2))
                v_score = max(0, 1 - cpa_v / (self.v_sep * 2))
                time_factor = max(0, 1 - t_cpa / self.lookahead)
                scores[idx] = h_score * v_score * (0.5 + 0.5 * time_factor)

        return torch.tensor(scores, dtype=torch.float32)

    def count_parameters(self) -> int:
        return 0

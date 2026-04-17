"""Per-track threat scoring.

Threat is a smooth function of (proximity to defended area, closing
speed, residual flight time, target classification confidence).  Used
both as a launch gate (must exceed ``threat_score_threshold``) and as
the prioritization key when more than one threat is present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThreatScorer:
    threshold: float = 0.62
    geofence_buffer_m: float = 200.0

    def score(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        class_confidence: float,
        defended_centre: np.ndarray,
        defended_radius_m: float,
    ) -> float:
        d = float(np.linalg.norm(position[:2] - defended_centre[:2]))
        prox = max(0.0, 1.0 - d / max(1.0, defended_radius_m))
        # closing speed component (positive = approaching defended centre)
        if d < 1e-3:
            closing = 0.0
        else:
            radial = (defended_centre[:2] - position[:2]) / d
            v2 = velocity[:2]
            closing = float(np.dot(v2, radial))
        speed_norm = max(0.0, min(1.0, closing / 60.0))
        cls_norm = max(0.0, min(1.0, class_confidence))
        # Weighted combination: a confirmed-track baseline, proximity dominates,
        # closing speed amplifies, and classification confidence trims tail.
        score = 0.30 + 0.40 * prox + 0.20 * speed_norm + 0.10 * cls_norm
        return float(min(1.0, max(0.0, score)))

    def is_launch_eligible(self, score: float) -> bool:
        return score >= self.threshold

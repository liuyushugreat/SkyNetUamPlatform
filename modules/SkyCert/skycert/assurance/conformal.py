"""Split-conformal prediction for multiclass UAM risk reasoning.

Two nonconformity scores are supported:

* ``lac`` — Least Ambiguous Classifier  (``s = 1 - p_y``), the classic
  score of Sadinle et al. (2019). Gives small sets under heavy skew but
  under-covers rare classes.
* ``aps`` — Adaptive Prediction Sets (Romano et al., 2020). Gives
  class-conditionally stable coverage at the cost of slightly larger
  sets, which is what we want for safety-critical risk reasoning.

Given a calibration set ``(X_cal, y_cal)`` and miscoverage level
``alpha``, ``ConformalRiskSet`` computes the empirical
``(1 - alpha)(1 + 1/n)``-quantile of calibration scores and, at test
time, returns the prediction set

    C(x) = { y : s(x, y) <= q_hat }.
"""

from __future__ import annotations

import numpy as np


class ConformalRiskSet:
    def __init__(self, alpha: float = 0.1, score: str = "aps") -> None:
        if score not in {"aps", "lac"}:
            raise ValueError(f"unknown score {score!r}")
        self.alpha = float(alpha)
        self.score = score
        self.q_hat: float | None = None
        self.num_classes: int | None = None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n, k = probs.shape
        order = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumulative = np.cumsum(sorted_probs, axis=1)
        rank_of_true = np.argmax(order == labels[:, None], axis=1)
        s = cumulative[np.arange(n), rank_of_true]
        return s

    @staticmethod
    def _aps_class_scores(probs: np.ndarray) -> np.ndarray:
        """Return an (n, k) matrix of APS scores for every candidate class."""
        n, k = probs.shape
        order = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumulative = np.cumsum(sorted_probs, axis=1)
        out = np.empty_like(probs)
        out[np.arange(n)[:, None], order] = cumulative
        return out

    @staticmethod
    def _lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return 1.0 - probs[np.arange(probs.shape[0]), labels]

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> float:
        n, k = probs.shape
        self.num_classes = k
        if self.score == "aps":
            s = self._aps_scores(probs, labels)
        else:
            s = self._lac_scores(probs, labels)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = float(min(level, 1.0))
        self.q_hat = float(np.quantile(s, level, method="higher"))
        return self.q_hat

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """Return a boolean mask ``(n, k)`` with ``True`` on included labels."""
        if self.q_hat is None or self.num_classes is None:
            raise RuntimeError("ConformalRiskSet.calibrate must be called first")
        if self.score == "aps":
            class_scores = self._aps_class_scores(probs)
        else:
            class_scores = 1.0 - probs
        mask = class_scores <= self.q_hat
        # Guard: always include the top-1 label so we never emit an empty set.
        top1 = np.argmax(probs, axis=1)
        mask[np.arange(mask.shape[0]), top1] = True
        return mask

    def coverage(self, probs: np.ndarray, labels: np.ndarray) -> float:
        mask = self.predict_sets(probs)
        return float(mask[np.arange(labels.shape[0]), labels].mean())

    def average_set_size(self, probs: np.ndarray) -> float:
        mask = self.predict_sets(probs)
        return float(mask.sum(axis=1).mean())

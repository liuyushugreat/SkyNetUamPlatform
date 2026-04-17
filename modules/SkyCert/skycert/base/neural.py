"""Neural risk scorer.

Kept intentionally lightweight (pure NumPy multinomial logistic regression)
so that experiments reproduce deterministically without a GPU. The
scorer emits *logits*, which are then combined with the symbolic engine.

When integrated into ``SkyNetUamPlatform``, this class can be replaced
by any callable mapping ``X -> logits`` with the same shape contract.
"""

from __future__ import annotations

import numpy as np

from ..utils import one_hot, softmax


class NeuralRiskScorer:
    def __init__(
        self,
        num_classes: int,
        l2: float = 5e-4,
        max_iter: int = 500,
        lr: float = 0.5,
        class_balanced: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.l2 = l2
        self.max_iter = max_iter
        self.lr = lr
        self.class_balanced = class_balanced
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralRiskScorer":
        n, d = X.shape
        k = self.num_classes
        rng = np.random.default_rng(0)
        self.W = 0.01 * rng.standard_normal((d, k))
        self.b = np.zeros(k)
        Y = one_hot(y, k)
        # Inverse-frequency class weights so the minority CRITICAL class
        # actually contributes to the gradient. Falls back to uniform
        # weighting when ``class_balanced=False``.
        counts = np.bincount(y, minlength=k).astype(np.float64)
        if self.class_balanced:
            weights = n / (k * np.maximum(counts, 1.0))
        else:
            weights = np.ones(k)
        sample_w = weights[y]
        sample_w = sample_w / sample_w.mean()
        for _ in range(self.max_iter):
            logits = X @ self.W + self.b
            probs = softmax(logits)
            diff = (probs - Y) * sample_w[:, None]
            gW = X.T @ diff / n + self.l2 * self.W
            gb = diff.mean(axis=0)
            self.W -= self.lr * gW
            self.b -= self.lr * gb
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.b is not None, "call fit() first"
        return X @ self.W + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

"""Neural risk scorer.

Two backbones are available, both pure NumPy so experiments reproduce
deterministically without a GPU:

* ``logistic`` (default): multinomial logistic regression. Used for the
  main body of the paper because it is transparent and fast.
* ``mlp``: two-layer multilayer perceptron with a ReLU hidden layer.
  Used to verify \\textsc{SkyCert}'s model-agnostic claim (Section 10)
  by replicating the main results with a non-linear backbone.

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
        model_type: str = "logistic",
        hidden: int = 64,
    ) -> None:
        self.num_classes = num_classes
        self.l2 = l2
        self.max_iter = max_iter
        self.lr = lr
        self.class_balanced = class_balanced
        self.model_type = model_type
        self.hidden = hidden
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.W1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.W2: np.ndarray | None = None
        self.b2: np.ndarray | None = None

    def _sample_weights(self, y: np.ndarray, k: int) -> np.ndarray:
        n = y.shape[0]
        counts = np.bincount(y, minlength=k).astype(np.float64)
        if self.class_balanced:
            weights = n / (k * np.maximum(counts, 1.0))
        else:
            weights = np.ones(k)
        sw = weights[y]
        return sw / sw.mean()

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> None:
        n, d = X.shape
        k = self.num_classes
        rng = np.random.default_rng(0)
        self.W = 0.01 * rng.standard_normal((d, k))
        self.b = np.zeros(k)
        Y = one_hot(y, k)
        for _ in range(self.max_iter):
            logits = X @ self.W + self.b
            probs = softmax(logits)
            diff = (probs - Y) * sw[:, None]
            gW = X.T @ diff / n + self.l2 * self.W
            gb = diff.mean(axis=0)
            self.W -= self.lr * gW
            self.b -= self.lr * gb

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> None:
        n, d = X.shape
        k = self.num_classes
        h = self.hidden
        rng = np.random.default_rng(0)
        # He-ish init for ReLU.
        self.W1 = rng.standard_normal((d, h)) * np.sqrt(2.0 / d)
        self.b1 = np.zeros(h)
        self.W2 = rng.standard_normal((h, k)) * np.sqrt(2.0 / h)
        self.b2 = np.zeros(k)
        Y = one_hot(y, k)
        lr = self.lr
        for _ in range(self.max_iter):
            z1 = X @ self.W1 + self.b1
            a1 = np.maximum(z1, 0.0)
            logits = a1 @ self.W2 + self.b2
            probs = softmax(logits)
            dz2 = (probs - Y) * sw[:, None]
            gW2 = a1.T @ dz2 / n + self.l2 * self.W2
            gb2 = dz2.mean(axis=0)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0).astype(np.float64)
            gW1 = X.T @ dz1 / n + self.l2 * self.W1
            gb1 = dz1.mean(axis=0)
            self.W1 -= lr * gW1
            self.b1 -= lr * gb1
            self.W2 -= lr * gW2
            self.b2 -= lr * gb2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralRiskScorer":
        k = self.num_classes
        sw = self._sample_weights(y, k)
        if self.model_type == "logistic":
            self._fit_logistic(X, y, sw)
        elif self.model_type == "mlp":
            self._fit_mlp(X, y, sw)
        else:
            raise ValueError(f"unknown model_type: {self.model_type!r}")
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        if self.model_type == "logistic":
            assert self.W is not None and self.b is not None, "call fit() first"
            return X @ self.W + self.b
        assert (
            self.W1 is not None and self.b1 is not None
            and self.W2 is not None and self.b2 is not None
        ), "call fit() first"
        a1 = np.maximum(X @ self.W1 + self.b1, 0.0)
        return a1 @ self.W2 + self.b2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

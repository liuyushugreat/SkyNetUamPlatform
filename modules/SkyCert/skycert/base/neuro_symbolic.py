"""Neuro-symbolic risk reasoner.

Combines a ``NeuralRiskScorer`` with a ``SymbolicRuleEngine`` into a
single object exposing ``predict_proba(X)`` and ``predict(X)``.  The
SkyCert assurance layer wraps around this object without needing to
know how either sub-component works.
"""

from __future__ import annotations

import numpy as np

from ..utils import softmax
from .neural import NeuralRiskScorer
from .symbolic import SymbolicRuleEngine


class NeuroSymbolicRiskReasoner:
    def __init__(
        self,
        neural: NeuralRiskScorer,
        symbolic: SymbolicRuleEngine,
        lambda_: float = 0.35,
    ) -> None:
        self.neural = neural
        self.symbolic = symbolic
        self.lambda_ = lambda_

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.neural.logits(X) + self.lambda_ * self.symbolic.logits(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(X), axis=-1)

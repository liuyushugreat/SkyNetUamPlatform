"""Lightweight symbolic rule engine.

Each rule is a predicate over a single feature plus a per-class logit
delta.  The engine sums deltas over all firing rules to produce a
symbolic logit vector that is added to the neural logits.

This mirrors the symbolic side of SkyKg / SkyFlow in spirit (a
knowledge-graph query returns a set of matching rules, each contributing
to the risk distribution) while remaining small enough to be audited
and attacked in a reproducible way.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import SymbolicRule


class SymbolicRuleEngine:
    def __init__(self, rules: Sequence[SymbolicRule], num_classes: int) -> None:
        self.rules = list(rules)
        self.num_classes = num_classes

    def logits(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros((n, self.num_classes), dtype=np.float64)
        if not self.rules:
            return out
        for r in self.rules:
            col = X[:, r.feature]
            if r.op == ">":
                fires = col > r.thr
            elif r.op == "<":
                fires = col < r.thr
            elif r.op == ">=":
                fires = col >= r.thr
            elif r.op == "<=":
                fires = col <= r.thr
            elif r.op == "==":
                fires = np.isclose(col, r.thr)
            else:
                raise ValueError(f"unsupported op {r.op!r}")
            delta = np.asarray(r.delta, dtype=np.float64)
            out += fires[:, None].astype(np.float64) * delta[None, :]
        return out

    def trace(self, x: np.ndarray) -> list[dict]:
        """Return a per-sample list of firing rules for audit artifacts."""
        events: list[dict] = []
        for idx, r in enumerate(self.rules):
            col = x[r.feature]
            fires: bool
            if r.op == ">":
                fires = bool(col > r.thr)
            elif r.op == "<":
                fires = bool(col < r.thr)
            elif r.op == ">=":
                fires = bool(col >= r.thr)
            elif r.op == "<=":
                fires = bool(col <= r.thr)
            elif r.op == "==":
                fires = bool(np.isclose(col, r.thr))
            else:
                raise ValueError(f"unsupported op {r.op!r}")
            if fires:
                events.append(
                    {
                        "rule_id": idx,
                        "feature": r.feature,
                        "op": r.op,
                        "thr": r.thr,
                        "delta": list(r.delta),
                    }
                )
        return events

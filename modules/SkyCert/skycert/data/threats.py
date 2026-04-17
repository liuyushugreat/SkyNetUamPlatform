"""Threat injection primitives.

These functions implement the adversary model described in Section 2 of the
paper: the attacker can either corrupt the symbolic rule base (``rule_flip``
or ``rule_inject``), perturb the input features inside a small L-inf ball
(``feature_attack``), or shift the covariate distribution at test time
(``covariate_shift``).  All injections return fresh data / rule structures
and never mutate their arguments.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Sequence

import numpy as np

from ..config import SymbolicRule


def corrupt_rules(
    rules: Sequence[SymbolicRule],
    strength: float,
    rng: np.random.Generator,
) -> list[SymbolicRule]:
    """Flip / perturb rule deltas with probability ~ ``strength``."""

    corrupted: list[SymbolicRule] = []
    for r in rules:
        if rng.random() < strength:
            delta = [-d + 0.1 * rng.standard_normal() for d in r.delta]
            corrupted.append(replace(r, delta=delta))
        else:
            corrupted.append(copy.deepcopy(r))
    return corrupted


def inject_rule_noise(
    rules: Sequence[SymbolicRule],
    strength: float,
    num_features: int,
    num_classes: int,
    rng: np.random.Generator,
) -> list[SymbolicRule]:
    """Add spurious rules that fire on unrelated features."""

    poisoned = [copy.deepcopy(r) for r in rules]
    num_new = max(1, int(round(strength * len(rules))))
    for _ in range(num_new):
        feature = int(rng.integers(0, num_features))
        op = str(rng.choice([">", "<"]))
        thr = float(rng.uniform(0.2, 0.8))
        delta = (rng.standard_normal(num_classes) * 0.4).tolist()
        poisoned.append(
            SymbolicRule(feature=feature, op=op, thr=thr, delta=delta)
        )
    return poisoned


def perturb_features(
    X: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Untargeted L-inf bounded perturbation used for sensor-spoof simulation."""
    noise = rng.uniform(-strength, strength, size=X.shape)
    return np.clip(X + noise, 0.0, 1.0)


def shift_covariates(
    X: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Push feature distribution toward the high-risk regime.

    We simulate a combination of realistic UAM-level stressors: a
    meteorological front (higher weather severity, higher humidity),
    rush-hour traffic (higher airspace density and vertiport congestion),
    fleet-wide battery fatigue (lower state-of-charge), plus a global
    variance inflation that reflects a degraded sensing stack.
    """
    bias = np.zeros(X.shape[1])
    bias[:5] = np.array([+1, +1, +1, -1, +1])
    # Global heteroscedastic noise captures a degraded sensing stack.
    het = 0.2 * strength * rng.standard_normal(X.shape)
    shifted = X + strength * 0.4 * bias[None, :] + het
    # Flip a subset of features into the opposite regime with probability
    # ``strength/2`` to simulate sensor-source substitution / failover.
    mask = rng.random(X.shape) < 0.5 * strength
    shifted = np.where(mask, 1.0 - shifted, shifted)
    return np.clip(shifted, 0.0, 1.0)


def apply_threat(
    kind: str,
    *,
    X: np.ndarray | None = None,
    rules: Sequence[SymbolicRule] | None = None,
    strength: float = 0.0,
    num_features: int | None = None,
    num_classes: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Dispatch to the correct primitive and return a dict with updates.

    Returns a mapping with keys among ``{"X", "rules"}``: only the pieces
    actually changed by the given threat are present, so callers can merge
    the result into the baseline run.
    """
    if rng is None:
        rng = np.random.default_rng()

    if kind == "none":
        return {}
    if kind == "rule_flip":
        assert rules is not None
        return {"rules": corrupt_rules(rules, strength, rng)}
    if kind == "rule_inject":
        assert rules is not None and num_features and num_classes
        return {
            "rules": inject_rule_noise(
                rules, strength, num_features, num_classes, rng
            )
        }
    if kind == "feature_attack":
        assert X is not None
        return {"X": perturb_features(X, strength, rng)}
    if kind == "covariate_shift":
        assert X is not None
        return {"X": shift_covariates(X, strength, rng)}
    raise ValueError(f"unknown threat kind: {kind!r}")

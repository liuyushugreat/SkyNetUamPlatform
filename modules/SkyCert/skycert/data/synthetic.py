"""Synthetic UAM risk dataset generator.

Features are normalised physical/operational quantities that a UAM
controller would have at hand for a single eVTOL operation request:

    f0: weather severity
    f1: airspace traffic density
    f2: vertiport congestion
    f3: battery state-of-charge
    f4: noise-sensitive area exposure
    f5: time of day (sin-encoded)
    f6: time of day (cos-encoded)
    f7-f15: auxiliary signals (wind, humidity, vertical rate, ...)

Labels are one of four ordinal risk levels
``{0: LOW, 1: MEDIUM, 2: HIGH, 3: CRITICAL}``.

The generator uses a latent score (affine combination of the five
safety-relevant features plus noise) pushed through ordered thresholds
so that the risk classes are mutually exclusive but correlated across
features, which makes it non-trivial for a pure rule-based or pure
neural baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UAMDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_classes: int


_LATENT_WEIGHTS = np.array(
    # Feature --> latent-risk contribution.
    [+1.2, +0.9, +0.7, -0.8, +0.6, +0.15, +0.10,
     +0.05, -0.05, +0.04, +0.04, -0.03, +0.02, +0.02, -0.02, +0.01]
)


def make_uam_dataset(
    num_train: int,
    num_calib: int,
    num_test: int,
    num_features: int = 16,
    num_classes: int = 4,
    class_prior: list[float] | None = None,
    seed: int = 0,
) -> UAMDataset:
    """Generate a train/calibration/test split of synthetic UAM operations."""

    rng = np.random.default_rng(seed)
    n = num_train + num_calib + num_test

    X = rng.beta(2.0, 2.0, size=(n, num_features))  # bounded in [0,1]
    # Impose weak correlation between time-of-day and weather for realism.
    hour = rng.uniform(0, 24, size=n)
    X[:, 5] = 0.5 * (np.sin(hour / 24 * 2 * np.pi) + 1)
    X[:, 6] = 0.5 * (np.cos(hour / 24 * 2 * np.pi) + 1)

    w = _LATENT_WEIGHTS[:num_features]
    latent = X @ w + 0.25 * rng.standard_normal(n)
    # Choose thresholds that roughly match the requested class prior.
    if class_prior is None:
        class_prior = [1.0 / num_classes] * num_classes
    prior = np.asarray(class_prior, dtype=np.float64)
    prior = prior / prior.sum()
    cum = np.cumsum(prior)[:-1]
    thresholds = np.quantile(latent, cum)
    y = np.digitize(latent, thresholds)

    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    i1 = num_train
    i2 = num_train + num_calib
    return UAMDataset(
        X_train=X[:i1], y_train=y[:i1],
        X_calib=X[i1:i2], y_calib=y[i1:i2],
        X_test=X[i2:], y_test=y[i2:],
        num_classes=num_classes,
    )

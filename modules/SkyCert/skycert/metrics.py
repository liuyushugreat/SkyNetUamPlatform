"""Evaluation metrics used across SkyCert experiments."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, num_bins: int = 15
) -> float:
    """Classic ECE (Guo et al., 2017) on top-1 confidences."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = labels.shape[0]
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def empirical_coverage(
    set_mask: np.ndarray, labels: np.ndarray
) -> float:
    return float(set_mask[np.arange(labels.shape[0]), labels].mean())


def average_set_size(set_mask: np.ndarray) -> float:
    return float(set_mask.sum(axis=1).mean())


def critical_error_rate(
    preds: np.ndarray, labels: np.ndarray, critical_class: int = 3
) -> float:
    """Rate of operations where the true risk is CRITICAL but the model
    output is anything lower. This is the safety-relevant false-negative
    rate for UAM risk reasoning."""
    mask = labels == critical_class
    if not mask.any():
        return 0.0
    return float((preds[mask] != critical_class).mean())


def abstain_error_rate(
    preds: np.ndarray,
    labels: np.ndarray,
    abstained: np.ndarray,
    critical_class: int = 3,
) -> float:
    """Critical-error rate *after* abstention (only counts non-abstained ops)."""
    keep = ~abstained
    if not keep.any():
        return 0.0
    return critical_error_rate(preds[keep], labels[keep], critical_class)


def per_class_coverage(
    set_mask: np.ndarray, labels: np.ndarray, num_classes: int
) -> dict[str, float]:
    """Conditional coverage ``P[y in C(x) | y = k]`` for every class ``k``.

    Reported as a dict ``{"class_k": float}`` so that downstream code can
    serialise it directly into ``metrics.json``. Missing classes (no
    samples with that label in the stream) are reported as ``None``.
    """
    out: dict[str, float | None] = {}
    for k in range(num_classes):
        mask = labels == k
        if not mask.any():
            out[f"class_{k}"] = None
        else:
            out[f"class_{k}"] = float(
                set_mask[np.arange(labels.shape[0])[mask], k].mean()
            )
    return out  # type: ignore[return-value]


def fp_fn_critical(
    preds: np.ndarray,
    labels: np.ndarray,
    abstained: np.ndarray,
    critical_class: int = 3,
) -> dict[str, float]:
    """False-positive / false-negative breakdown for the CRITICAL class.

    FN (``critical_fn``): true label is CRITICAL but the non-abstained
    prediction is something lower. This is the safety-relevant miss rate.
    FP (``critical_fp``): true label is non-CRITICAL but the non-abstained
    prediction is CRITICAL. This measures over-cautious mislabelling.
    """
    keep = ~abstained
    kept_preds = preds[keep]
    kept_labels = labels[keep]

    crit_mask = kept_labels == critical_class
    noncrit_mask = ~crit_mask

    if crit_mask.any():
        critical_fn = float((kept_preds[crit_mask] != critical_class).mean())
    else:
        critical_fn = 0.0
    if noncrit_mask.any():
        critical_fp = float((kept_preds[noncrit_mask] == critical_class).mean())
    else:
        critical_fp = 0.0

    # Overall retained-sample misclassification rate as a sanity anchor.
    if kept_labels.size:
        retained_error = float((kept_preds != kept_labels).mean())
    else:
        retained_error = 0.0

    return {
        "critical_fn": critical_fn,
        "critical_fp": critical_fp,
        "retained_error": retained_error,
        "retained_fraction": float(keep.mean()),
    }


def detection_metrics(
    alerts: np.ndarray, change_point: int | None
) -> dict[str, float]:
    """Summary statistics for streaming drift detection.

    Parameters
    ----------
    alerts:
        Boolean array of length ``T`` with ``True`` on alert steps.
    change_point:
        Index at which the distribution actually changes. If ``None``,
        the stream is IID and all alerts are false positives.
    """
    alerts = np.asarray(alerts, dtype=bool)
    T = alerts.shape[0]
    if change_point is None:
        return {
            "false_alarm_rate": float(alerts.mean()) if T else 0.0,
            "detection_delay": float("inf"),
            "detected": False,
        }
    pre = alerts[:change_point]
    post = alerts[change_point:]
    if post.any():
        first = int(np.argmax(post))
        return {
            "false_alarm_rate": float(pre.mean()) if pre.size else 0.0,
            "detection_delay": float(first),
            "detected": True,
        }
    return {
        "false_alarm_rate": float(pre.mean()) if pre.size else 0.0,
        "detection_delay": float("inf"),
        "detected": False,
    }

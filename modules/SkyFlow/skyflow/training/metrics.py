"""Conflict detection evaluation metrics.

CDR (recall), FAR (1 - precision), F1, and inference latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class MetricResult:
    cdr: float          # Conflict Detection Rate = recall
    far: float          # False Alert Rate = 1 - precision
    f1: float
    precision: float
    recall: float
    latency_ms: float   # 95th-percentile wall-clock inference time
    latency_mean_ms: float
    num_pairs: int
    num_positives: int


class ConflictMetrics:
    """Accumulates predictions across batches and computes final metrics."""

    def __init__(self, threshold: float = 0.42):
        self.threshold = threshold
        self.all_preds: List[np.ndarray] = []
        self.all_labels: List[np.ndarray] = []
        self.latencies: List[float] = []

    def reset(self):
        self.all_preds.clear()
        self.all_labels.clear()
        self.latencies.clear()

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        latency_ms: Optional[float] = None,
    ):
        self.all_preds.append(preds.detach().cpu().numpy())
        self.all_labels.append(labels.detach().cpu().numpy())
        if latency_ms is not None:
            self.latencies.append(latency_ms)

    def compute(self) -> MetricResult:
        preds = np.concatenate(self.all_preds)
        labels = np.concatenate(self.all_labels)

        predicted_pos = preds >= self.threshold
        actual_pos = labels >= 0.5

        tp = np.sum(predicted_pos & actual_pos)
        fp = np.sum(predicted_pos & ~actual_pos)
        fn = np.sum(~predicted_pos & actual_pos)
        tn = np.sum(~predicted_pos & ~actual_pos)

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        far = fp / max(tp + fp, 1)

        if self.latencies:
            lat_95 = float(np.percentile(self.latencies, 95))
            lat_mean = float(np.mean(self.latencies))
        else:
            lat_95 = 0.0
            lat_mean = 0.0

        return MetricResult(
            cdr=float(recall),
            far=float(far),
            f1=float(f1),
            precision=float(precision),
            recall=float(recall),
            latency_ms=lat_95,
            latency_mean_ms=lat_mean,
            num_pairs=len(preds),
            num_positives=int(np.sum(actual_pos)),
        )


class RegimeMetrics:
    """Compute per-regime (easy/hard) CDR and FAR.

    Per paper Section 6.2: easy = TTC > 45s, pairwise, benign weather;
    hard = TTC <= 45s, multi-aircraft, or high-wind-variance cells.
    """

    def __init__(self, threshold: float = 0.42, ttc_boundary: float = 45.0):
        self.threshold = threshold
        self.ttc_boundary = ttc_boundary
        self.easy_preds: List[np.ndarray] = []
        self.easy_labels: List[np.ndarray] = []
        self.hard_preds: List[np.ndarray] = []
        self.hard_labels: List[np.ndarray] = []

    def reset(self):
        self.easy_preds.clear()
        self.easy_labels.clear()
        self.hard_preds.clear()
        self.hard_labels.clear()

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        is_hard: torch.Tensor,
    ):
        """is_hard: (N,) bool tensor indicating hard conflict regime."""
        p = preds.detach().cpu().numpy()
        l = labels.detach().cpu().numpy()
        h = is_hard.detach().cpu().numpy().astype(bool)
        if h.any():
            self.hard_preds.append(p[h])
            self.hard_labels.append(l[h])
        easy_mask = ~h
        if easy_mask.any():
            self.easy_preds.append(p[easy_mask])
            self.easy_labels.append(l[easy_mask])

    def compute(self) -> dict:
        result = {}
        for regime, p_list, l_list in [
            ("easy", self.easy_preds, self.easy_labels),
            ("hard", self.hard_preds, self.hard_labels),
        ]:
            if not p_list:
                result[regime] = {"cdr": 0.0, "far": 0.0}
                continue
            preds = np.concatenate(p_list)
            labels = np.concatenate(l_list)
            predicted_pos = preds >= self.threshold
            actual_pos = labels >= 0.5
            tp = np.sum(predicted_pos & actual_pos)
            fp = np.sum(predicted_pos & ~actual_pos)
            fn = np.sum(~predicted_pos & actual_pos)
            recall = tp / max(tp + fn, 1)
            far = fp / max(tp + fp, 1)
            result[regime] = {"cdr": float(recall), "far": float(far)}
        return result


def bonferroni_ttest(
    trgat_cdrs: List[float],
    baseline_cdrs: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict[str, dict]:
    """Paired two-sided t-test with Bonferroni correction (Table 5 in paper)."""
    from scipy import stats

    n_comparisons = len(baseline_cdrs)
    corrected_alpha = alpha / max(n_comparisons, 1)
    results = {}

    trgat = np.array(trgat_cdrs)
    for name, bl_cdrs in baseline_cdrs.items():
        bl = np.array(bl_cdrs)
        delta = trgat - bl
        mean_delta = float(np.mean(delta))
        if len(delta) > 1 and np.std(delta) > 1e-12:
            t_stat, p_val = stats.ttest_rel(trgat, bl)
        else:
            t_stat, p_val = float("inf"), 0.0
        results[name] = {
            "delta_cdr": mean_delta,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < corrected_alpha),
            "corrected_alpha": corrected_alpha,
        }
    return results


class LatencyTimer:
    """Context manager for measuring inference latency."""

    def __init__(self):
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0

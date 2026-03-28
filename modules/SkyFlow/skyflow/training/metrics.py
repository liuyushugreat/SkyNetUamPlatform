"""Conflict detection evaluation metrics.

CDR (recall), FAR (1 - precision), F1, and inference latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

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

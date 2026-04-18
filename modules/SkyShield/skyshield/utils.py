"""Small pure-Python helpers (stats, geometry, RNG)."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import math

import numpy as np


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def summarize_latency(samples: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def euclid_km(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def dataclass_to_json(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: dataclass_to_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: dataclass_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

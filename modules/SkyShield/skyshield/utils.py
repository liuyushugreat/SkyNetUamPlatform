"""Shared helpers: deterministic RNG, JSON-safe dumping, percentile macros."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_json(obj: Any) -> Any:
    """Convert numpy / math.inf / tuples so ``json.dumps`` never raises."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


def dump_json(path: str | os.PathLike, data: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(safe_json(data), f, indent=2)


def load_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentiles(values: Iterable[float], ps: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {f"p{int(p*100)}": float("nan") for p in ps}
    return {f"p{int(p*100)}": float(np.percentile(arr, p * 100.0)) for p in ps}


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def deterministic_hash(s: str) -> int:
    """Stable 64-bit hash (no PYTHONHASHSEED dependency)."""
    h = 1469598103934665603
    for c in s.encode("utf-8"):
        h = (h ^ c) * 1099511628211
        h &= (1 << 64) - 1
    return h

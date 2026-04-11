"""Batch metric computation utilities for SkyGov experiments."""

from __future__ import annotations

import re
from typing import Dict, List, Set


def compute_rar(text: str) -> float:
    """Rule Adherence Rate: fraction of assertion sentences citing a rule."""
    sentences = [s.strip() for s in re.split(r"[。\n]", text) if len(s.strip()) > 5]
    if not sentences:
        return 0.0
    pattern = re.compile(r"REG-[A-Z]+-\d+")
    backed = sum(1 for s in sentences if pattern.search(s))
    return backed / len(sentences)


def compute_lec(cited: List[str], relevant: Set[str]) -> float:
    """Legal Entity Coverage: fraction of relevant rules cited."""
    if not relevant:
        return 1.0
    return len(set(cited) & relevant) / len(relevant)


def compute_ucr(text: str) -> float:
    """Unsupported Claim Rate: fraction of assertion sentences without rule backing."""
    return 1.0 - compute_rar(text)


def compute_all(text: str, cited: List[str], relevant: Set[str]) -> Dict[str, float]:
    return {
        "rar": round(compute_rar(text), 4),
        "lec": round(compute_lec(cited, relevant), 4),
        "ucr": round(compute_ucr(text), 4),
    }

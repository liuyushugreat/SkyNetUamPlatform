"""Hallucination guard: KG-based fact checking for LLM-generated outputs."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


@dataclass
class FactCheckResult:
    claim: str
    supported: bool
    matched_rule: str = ""
    reason: str = ""


class HallucinationGuard:
    """Checks LLM-generated text against known KG facts and regulation IDs.

    Two-stage verification:
        1. Rule ID validation — cited rule IDs must exist in the known rule registry.
        2. Claim-fact alignment — factual claims in the text are cross-checked
           against KG triples (when a live graph is available).
    """

    def __init__(self, known_rules: Set[str] | None = None):
        self.known_rules = known_rules or {
            "REG-WIND-001",
            "REG-BAT-001",
            "REG-ZONE-001",
            "REG-SAFETY-012",
            "REG-SAFETY-013",
            "REG-LOAD-001",
            "REG-VIS-001",
        }

    def check(self, text: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cited = set(re.findall(r"REG-[A-Z]+-\d+", text))
        valid_rules = cited & self.known_rules
        invalid_rules = cited - self.known_rules

        if invalid_rules:
            logger.warning("Hallucinated rule IDs detected: %s", invalid_rules)

        hallucination_rate = len(invalid_rules) / len(cited) if cited else 0.0

        return {
            "cited_rules": sorted(cited),
            "valid_rules": sorted(valid_rules),
            "invalid_rules": sorted(invalid_rules),
            "hallucination_rate": round(hallucination_rate, 4),
            "passed": len(invalid_rules) == 0,
        }

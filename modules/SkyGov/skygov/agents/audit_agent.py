"""AuditAgent: real-time explanation quality scoring and hallucination interception."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .base_agent import BaseAgent, AgentResult, AgentVerdict, TraceEntry


class AuditAgent(BaseAgent):
    """Scores the quality of explanations using RAR/LEC/UCR metrics.

    Metrics (from SkyKG, KSEM 2026):
        - RAR (Rule Adherence Rate): fraction of claims backed by cited rules.
        - LEC (Legal Entity Coverage): fraction of relevant regulations cited.
        - UCR (Unsupported Claim Rate): fraction of claims without rule support.

    If quality falls below configured thresholds, the agent can trigger
    re-retrieval, re-generation, or human escalation.
    """

    name = "audit"

    def __init__(self, config=None, known_rule_ids: Set[str] | None = None):
        super().__init__(config)
        self.known_rule_ids = known_rule_ids or {
            "REG-WIND-001",
            "REG-BAT-001",
            "REG-ZONE-001",
            "REG-ALT-001",
            "REG-VIS-001",
            "REG-LOAD-001",
            "REG-SPEED-001",
            "REG-TEMP-001",
            "REG-SAFETY-012",
            "REG-SAFETY-013",
        }
        rar_thr = 0.8
        lec_thr = 0.6
        ucr_thr = 0.1
        if config:
            rar_thr = getattr(config, "rar_threshold", rar_thr)
            lec_thr = getattr(config, "lec_threshold", lec_thr)
            ucr_thr = getattr(config, "ucr_threshold", ucr_thr)
        self.rar_threshold = rar_thr
        self.lec_threshold = lec_thr
        self.ucr_threshold = ucr_thr

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        explanation_text = context.get("explanation", "")
        cited_rules_from_agent: List[str] = context.get("cited_rules", [])
        relevant_rules: Set[str] = set(context.get("relevant_rules", []))
        if not relevant_rules:
            relevant_rules = self.known_rule_ids

        rar = self._compute_rar(explanation_text, cited_rules_from_agent)
        lec = self._compute_lec(cited_rules_from_agent, relevant_rules)
        ucr = self._compute_ucr(explanation_text, cited_rules_from_agent)

        passed = (
            rar >= self.rar_threshold
            and lec >= self.lec_threshold
            and ucr <= self.ucr_threshold
        )

        traces = [
            TraceEntry(
                step="quality_audit",
                source=self.name,
                detail=(
                    f"RAR={rar:.2f} (thr={self.rar_threshold}), "
                    f"LEC={lec:.2f} (thr={self.lec_threshold}), "
                    f"UCR={ucr:.2f} (thr={self.ucr_threshold}) → "
                    f"{'PASS' if passed else 'FAIL'}"
                ),
            )
        ]

        return AgentResult(
            agent_name=self.name,
            verdict=AgentVerdict.SAFE if passed else AgentVerdict.UNCERTAIN,
            confidence=rar,
            payload={
                "rar": round(rar, 4),
                "lec": round(lec, 4),
                "ucr": round(ucr, 4),
                "passed": passed,
                "recommendation": "accept" if passed else "re_retrieve",
            },
            traces=traces,
        )

    def _compute_rar(self, text: str, cited_rules: List[str]) -> float:
        """Rule Adherence Rate: fraction of assertion sentences that cite a rule."""
        sentences = [s.strip() for s in re.split(r"[。\n]", text) if len(s.strip()) > 5]
        if not sentences:
            return 0.0
        rule_pattern = re.compile(r"REG-[A-Z]+-\d+")
        backed = sum(1 for s in sentences if rule_pattern.search(s))
        return backed / len(sentences)

    def _compute_lec(self, cited: List[str], relevant: Set[str]) -> float:
        """Legal Entity Coverage: fraction of relevant rules that are cited."""
        if not relevant:
            return 1.0
        cited_set = set(cited)
        return len(cited_set & relevant) / len(relevant)

    def _compute_ucr(self, text: str, cited_rules: List[str]) -> float:
        """Unsupported Claim Rate: fraction of assertion sentences without rule backing."""
        sentences = [s.strip() for s in re.split(r"[。\n]", text) if len(s.strip()) > 5]
        if not sentences:
            return 0.0
        rule_pattern = re.compile(r"REG-[A-Z]+-\d+")
        unsupported = sum(1 for s in sentences if not rule_pattern.search(s))
        return unsupported / len(sentences)

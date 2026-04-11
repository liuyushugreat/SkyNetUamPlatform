"""Hierarchical Trust Negotiation Protocol for multi-agent decision fusion.

Implements a three-layer trust protocol:
  Layer 1 (Veto):    Any agent with veto power returning VIOLATION → immediate reject.
  Layer 2 (Quality): Audit RAR below threshold → degrade to "uncertain".
  Layer 3 (Voting):  Weighted confidence voting among reasoning agents.

The weighted fusion formula:
  w_risk = Σ_i (α_i · c_i · I[v_i ∈ {risk, uncertain}]) / Σ_i (α_i · c_i)

where α_i is the preset weight for agent i, c_i is its self-reported confidence,
and I[·] is the indicator function for risk-positive verdicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..agents.base_agent import AgentResult, AgentVerdict

logger = logging.getLogger(__name__)


DEFAULT_AGENT_WEIGHTS = {
    "risk_assessment": 0.6,
    "explanation": 0.4,
}


@dataclass
class TrustVote:
    agent_name: str
    verdict: AgentVerdict
    confidence: float
    quality_score: float = 1.0
    weight: float = 1.0


class TrustProtocol:
    """Aggregates agent outputs into a final governance decision.

    Three-layer hierarchical trust protocol:
        Layer 1 — Veto:    Deterministic short-circuit on hard-rule violation.
        Layer 2 — Quality: Quality-gated downgrade when audit fails.
        Layer 3 — Voting:  Weighted confidence fusion for final verdict.
    """

    def __init__(
        self,
        veto_agents: Optional[List[str]] = None,
        quality_threshold: float = 0.8,
        risk_threshold: float = 0.5,
        agent_weights: Optional[Dict[str, float]] = None,
    ):
        self.veto_agents = set(veto_agents or ["compliance"])
        self.quality_threshold = quality_threshold
        self.risk_threshold = risk_threshold
        self.agent_weights = agent_weights or dict(DEFAULT_AGENT_WEIGHTS)

    def aggregate(self, results: Dict[str, AgentResult]) -> Dict:
        """Fuse multi-agent results via the hierarchical trust protocol.

        Returns a dict with final_verdict, reason, confidence, quality_check,
        action, and the full voting details for audit traceability.
        """
        votes = self._collect_votes(results)

        # --- Layer 1: Veto ---
        veto_decision = self._layer_veto(votes)
        if veto_decision is not None:
            return veto_decision

        # --- Layer 2: Quality Gate ---
        quality_decision = self._layer_quality_gate(votes)
        if quality_decision is not None:
            return quality_decision

        # --- Layer 3: Weighted Voting ---
        return self._layer_weighted_voting(votes)

    def _collect_votes(self, results: Dict[str, AgentResult]) -> List[TrustVote]:
        votes: List[TrustVote] = []
        for name, result in results.items():
            if not isinstance(result, AgentResult):
                continue
            quality = 1.0
            if name == "audit" and "rar" in result.payload:
                quality = result.payload["rar"]
            weight = self.agent_weights.get(name, 0.0)
            votes.append(
                TrustVote(
                    agent_name=name,
                    verdict=result.verdict,
                    confidence=result.confidence,
                    quality_score=quality,
                    weight=weight,
                )
            )
        return votes

    def _layer_veto(self, votes: List[TrustVote]) -> Optional[Dict]:
        """Layer 1: deterministic veto by privileged agents."""
        for vote in votes:
            if vote.agent_name in self.veto_agents and vote.verdict in (
                AgentVerdict.VIOLATION,
                AgentVerdict.VETO,
            ):
                logger.warning(
                    "VETO triggered by %s (verdict=%s)",
                    vote.agent_name, vote.verdict.value,
                )
                return {
                    "final_verdict": AgentVerdict.VIOLATION.value,
                    "reason": f"Vetoed by {vote.agent_name}",
                    "confidence": 1.0,
                    "quality_check": "N/A (vetoed)",
                    "action": "reject",
                    "layer": "veto",
                    "voting_details": None,
                }
        return None

    def _layer_quality_gate(self, votes: List[TrustVote]) -> Optional[Dict]:
        """Layer 2: quality-gated downgrade when audit fails."""
        audit_vote = next((v for v in votes if v.agent_name == "audit"), None)
        if audit_vote and audit_vote.quality_score < self.quality_threshold:
            return {
                "final_verdict": AgentVerdict.UNCERTAIN.value,
                "reason": (
                    f"Audit quality {audit_vote.quality_score:.2f} "
                    f"< threshold {self.quality_threshold}"
                ),
                "confidence": audit_vote.quality_score,
                "quality_check": "failed",
                "action": "re_retrieve",
                "layer": "quality_gate",
                "voting_details": None,
            }
        return None

    def _layer_weighted_voting(self, votes: List[TrustVote]) -> Dict:
        """Layer 3: weighted confidence fusion.

        w_risk = Σ(α_i * c_i * I_risk(v_i)) / Σ(α_i * c_i)
        where I_risk(v) = 1 if v ∈ {risk, uncertain}, else 0.
        """
        reasoning_votes = [
            v for v in votes
            if v.agent_name not in ("audit", "compliance") and v.weight > 0
        ]

        if not reasoning_votes:
            return {
                "final_verdict": AgentVerdict.SAFE.value,
                "reason": "No risk signals from any agent",
                "confidence": 1.0,
                "quality_check": "passed",
                "action": "approve",
                "layer": "voting",
                "voting_details": {"weighted_risk": 0.0, "votes": []},
            }

        total_weight = sum(v.weight * v.confidence for v in reasoning_votes)
        if total_weight == 0:
            weighted_risk = 0.0
        else:
            risk_verdicts = (AgentVerdict.RISK, AgentVerdict.UNCERTAIN)
            weighted_risk = sum(
                v.weight * v.confidence * (1.0 if v.verdict in risk_verdicts else 0.0)
                for v in reasoning_votes
            ) / total_weight

        if weighted_risk > self.risk_threshold:
            final = AgentVerdict.RISK
            action = "conditional_approve"
        else:
            final = AgentVerdict.SAFE
            action = "approve"

        vote_details = [
            {
                "agent": v.agent_name,
                "verdict": v.verdict.value,
                "confidence": round(v.confidence, 4),
                "weight": v.weight,
                "contribution": round(v.weight * v.confidence, 4),
            }
            for v in reasoning_votes
        ]

        return {
            "final_verdict": final.value,
            "reason": f"Weighted risk score: {weighted_risk:.4f}",
            "confidence": round(1.0 - weighted_risk, 4),
            "quality_check": "passed",
            "action": action,
            "layer": "voting",
            "voting_details": {
                "weighted_risk": round(weighted_risk, 4),
                "risk_threshold": self.risk_threshold,
                "votes": vote_details,
            },
        }

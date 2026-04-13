"""Explanation rules: generate structured, RDF-serializable explanations
for governance decisions, valuation scores, and promotion outcomes.

Each explanation is a structured object that can be injected into the KG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF

from ..rdf.namespaces import SKYRWA, SKYRWA_INST
from ..models.asset_unit import FlightAssetUnit
from ..models.valuation import ValuationResultV2


@dataclass
class ExplanationEntry:
    dimension: str
    score: float
    impact: str  # "positive" | "negative" | "neutral"
    detail: str


@dataclass
class Explanation:
    subject_id: str
    subject_type: str
    conclusion: str
    factors: List[ExplanationEntry] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"[{self.subject_type}] {self.subject_id}: {self.conclusion}"]
        for f in self.factors:
            lines.append(f"  - {f.dimension}: {f.score:.2f} ({f.impact}) — {f.detail}")
        return "\n".join(lines)


class ExplanationBuilder:
    """Build structured explanations for pipeline decisions."""

    @staticmethod
    def explain_valuation(unit: FlightAssetUnit) -> Explanation:
        """Explain why an asset received its valuation score."""
        val = unit.valuation_result
        if val is None:
            return Explanation(
                subject_id=unit.asset_unit_id,
                subject_type="valuation",
                conclusion="No valuation performed yet.",
            )
        factors: List[ExplanationEntry] = []
        for name, score in val.breakdown.items():
            if score >= 0.7:
                impact = "positive"
                detail = f"{name} scored high ({score:.2f}), boosting value"
            elif score <= 0.3:
                impact = "negative"
                detail = f"{name} scored low ({score:.2f}), reducing value"
            else:
                impact = "neutral"
                detail = f"{name} scored moderate ({score:.2f})"
            factors.append(ExplanationEntry(
                dimension=name, score=score, impact=impact, detail=detail,
            ))
        return Explanation(
            subject_id=unit.asset_unit_id,
            subject_type="valuation",
            conclusion=f"Estimated value: {val.estimated_value:.2f} {val.currency} "
                        f"(confidence: {val.confidence:.2f})",
            factors=factors,
        )

    @staticmethod
    def explain_governance(unit: FlightAssetUnit) -> Explanation:
        """Explain governance outcome (tradability, desensitization, etc.)."""
        rp = unit.rights_profile
        factors: List[ExplanationEntry] = []

        factors.append(ExplanationEntry(
            dimension="compliance",
            score=unit.compliance_score,
            impact="positive" if unit.compliance_score >= 0.8 else "negative",
            detail=f"Compliance score: {unit.compliance_score:.2f}",
        ))
        factors.append(ExplanationEntry(
            dimension="risk",
            score=unit.risk_score,
            impact="negative" if unit.risk_score > 0.5 else "positive",
            detail=f"Risk score: {unit.risk_score:.2f}",
        ))

        if rp:
            tradable = rp.tradable
            desen = rp.desensitization_required
            if tradable and not desen:
                conclusion = "Tradable: the asset passed governance and can be licensed."
            elif tradable and desen:
                conclusion = "Tradable after desensitization: raw data must be anonymized first."
            else:
                conclusion = "Non-transferable: governance rules prevent external sharing."
        else:
            conclusion = "No rights profile assigned — governance not yet applied."

        return Explanation(
            subject_id=unit.asset_unit_id,
            subject_type="governance",
            conclusion=conclusion,
            factors=factors,
        )

    @staticmethod
    def explain_promotion_eligibility(unit: FlightAssetUnit) -> Explanation:
        """Explain whether a single asset is eligible for product promotion."""
        factors: List[ExplanationEntry] = []
        eligible = True
        reasons: List[str] = []

        if unit.compliance_score < 0.7:
            eligible = False
            reasons.append("compliance below 0.7")
        factors.append(ExplanationEntry(
            dimension="compliance_threshold",
            score=unit.compliance_score,
            impact="positive" if unit.compliance_score >= 0.7 else "negative",
            detail=f"Compliance {unit.compliance_score:.2f} vs threshold 0.70",
        ))

        if unit.data_quality_score < 0.5:
            eligible = False
            reasons.append("data quality below 0.5")
        factors.append(ExplanationEntry(
            dimension="quality_threshold",
            score=unit.data_quality_score,
            impact="positive" if unit.data_quality_score >= 0.5 else "negative",
            detail=f"Quality {unit.data_quality_score:.2f} vs threshold 0.50",
        ))

        rp = unit.rights_profile
        if rp is None or not rp.tradable:
            eligible = False
            reasons.append("not tradable per rights profile")
        factors.append(ExplanationEntry(
            dimension="tradability",
            score=1.0 if (rp and rp.tradable) else 0.0,
            impact="positive" if (rp and rp.tradable) else "negative",
            detail="Tradable" if (rp and rp.tradable) else "Not tradable",
        ))

        conclusion = ("Eligible for product promotion"
                       if eligible
                       else f"Not eligible: {'; '.join(reasons)}")

        return Explanation(
            subject_id=unit.asset_unit_id,
            subject_type="promotion",
            conclusion=conclusion,
            factors=factors,
        )

    @staticmethod
    def inject_explanation(graph: Graph, explanation: Explanation) -> URIRef:
        """Serialize an Explanation into the RDF graph."""
        subj = SKYRWA_INST[f"explanation:{explanation.subject_id}:{explanation.subject_type}"]
        graph.add((subj, RDF.type, SKYRWA.ValuationExplanation))
        graph.add((subj, SKYRWA["conclusion"], Literal(explanation.conclusion)))
        graph.add((subj, SKYRWA["subjectType"], Literal(explanation.subject_type)))
        for f in explanation.factors:
            factor = BNode()
            graph.add((subj, SKYRWA.hasValuationFactor, factor))
            graph.add((factor, RDF.type, SKYRWA.ValuationFactor))
            graph.add((factor, SKYRWA["factorName"], Literal(f.dimension)))
            graph.add((factor, SKYRWA["factorScore"],
                        Literal(f.score, datatype="http://www.w3.org/2001/XMLSchema#float")))
            graph.add((factor, SKYRWA["impact"], Literal(f.impact)))
            graph.add((factor, SKYRWA["detail"], Literal(f.detail)))
        return subj

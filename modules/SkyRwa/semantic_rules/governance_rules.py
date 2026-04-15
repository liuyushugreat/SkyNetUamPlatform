"""Semantic governance rules expressed as SPARQL CONSTRUCT patterns.

These rules complement the Python-based GovernanceEngine by providing
machine-readable, auditable rule definitions that can be serialized
into the knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from rdflib import Graph, Literal, URIRef, BNode
from rdflib.namespace import RDF, XSD

from ..rdf.namespaces import SKYRWA, SKYRWA_INST


@dataclass
class GovernanceRuleResult:
    rule_id: str
    rule_label: str
    affected_assets: List[str] = field(default_factory=list)
    explanation: str = ""


class GovernanceRuleEngine:
    """Execute governance rules as SPARQL queries on the KG."""

    RULE_NON_TRANSFERABLE = """
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

    SELECT ?asset ?flightId ?compliance ?risk
    WHERE {
        ?asset a skyrwa:AssetCandidate ;
               skyrwa:flightId ?flightId ;
               skyrwa:complianceScore ?compliance ;
               skyrwa:riskScore ?risk .
        FILTER (?compliance < 0.5 || ?risk > 0.8)
    }
    """

    RULE_DESENSITIZATION_REQUIRED = """
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

    SELECT ?asset ?flightId
    WHERE {
        ?asset a skyrwa:AssetCandidate ;
               skyrwa:flightId ?flightId ;
               skyrwa:hasRightsProfile ?rp .
        ?rp skyrwa:requiresDesensitization "true"^^xsd:boolean .
        ?rp skyrwa:isTradable "true"^^xsd:boolean .
    }
    """

    RULE_MISSING_DIGEST = """
    PREFIX skyrwa: <urn:skyrwa:ontology#>

    SELECT ?evidence ?flightId
    WHERE {
        ?evidence a skyrwa:FlightEvidence ;
                  skyrwa:flightId ?flightId .
        FILTER NOT EXISTS { ?evidence skyrwa:hasDigest ?d }
    }
    """

    @staticmethod
    def run_all(graph: Graph) -> List[GovernanceRuleResult]:
        """Execute all governance rules and return results."""
        results: List[GovernanceRuleResult] = []

        rows = list(graph.query(GovernanceRuleEngine.RULE_NON_TRANSFERABLE))
        results.append(GovernanceRuleResult(
            rule_id="GOV-001",
            rule_label="Non-transferable due to low compliance or high risk",
            affected_assets=[str(r[1]) for r in rows],
            explanation=f"{len(rows)} asset(s) fail compliance/risk threshold",
        ))

        rows = list(graph.query(GovernanceRuleEngine.RULE_DESENSITIZATION_REQUIRED))
        results.append(GovernanceRuleResult(
            rule_id="GOV-002",
            rule_label="Tradable but requires desensitization",
            affected_assets=[str(r[1]) for r in rows],
            explanation=f"{len(rows)} asset(s) marked tradable but need desensitization first",
        ))

        rows = list(graph.query(GovernanceRuleEngine.RULE_MISSING_DIGEST))
        results.append(GovernanceRuleResult(
            rule_id="GOV-003",
            rule_label="Evidence missing digest hash",
            affected_assets=[str(r[1]) for r in rows],
            explanation=f"{len(rows)} evidence record(s) lack a content digest",
        ))

        return results

    @staticmethod
    def inject_decisions(graph: Graph, results: List[GovernanceRuleResult]) -> None:
        """Materialize governance decisions as triples in the graph.

        Each affected asset gets its own GovernanceDecision node linked via
        ``skyrwa:appliedToAsset``, enabling per-asset SPARQL audit queries.
        """
        for r in results:
            if not r.affected_assets:
                continue
            for idx, flight_id in enumerate(r.affected_assets):
                decision = SKYRWA_INST[
                    f"governance_decision:{r.rule_id}:{flight_id}"
                ]
                graph.add((decision, RDF.type, SKYRWA.GovernanceDecision))
                graph.add((decision, SKYRWA["ruleId"], Literal(r.rule_id)))
                graph.add(
                    (decision, SKYRWA["ruleLabel"], Literal(r.rule_label))
                )
                graph.add(
                    (decision, SKYRWA["explanation"], Literal(r.explanation))
                )
                asset_candidates = list(graph.subjects(
                    SKYRWA["flightId"], Literal(flight_id)
                ))
                for asset_uri in asset_candidates:
                    if (asset_uri, RDF.type, SKYRWA.AssetCandidate) in graph \
                            or (asset_uri, RDF.type, SKYRWA.FlightEvidence) in graph:
                        graph.add(
                            (decision, SKYRWA["appliedToAsset"], asset_uri)
                        )

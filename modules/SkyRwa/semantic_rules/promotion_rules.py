"""Promotion rules: determine when asset candidates can be promoted to
governed data products.

Rules are expressed as SPARQL queries for transparency and auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from rdflib import Graph, Literal
from rdflib.namespace import RDF

from ..rdf.namespaces import SKYRWA, SKYRWA_INST


@dataclass
class PromotionCandidate:
    product_type: str
    candidate_iris: List[str] = field(default_factory=list)
    flight_ids: List[str] = field(default_factory=list)
    eligible: bool = False
    reason: str = ""


class PromotionRuleEngine:
    """Evaluate whether a set of asset candidates can be promoted to a
    GovernedDataProduct."""

    MIN_CANDIDATES = 3
    MIN_QUALITY = 0.5
    MIN_COMPLIANCE = 0.7

    QUERY_PROMOTABLE_BY_CLASS = """
    PREFIX skyrwa: <https://w3id.org/skyrwa#>
    PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

    SELECT ?assetClass (COUNT(?asset) AS ?cnt)
           (AVG(?quality) AS ?avgQuality) (AVG(?compliance) AS ?avgCompliance)
    WHERE {{
        ?asset a skyrwa:AssetCandidate ;
               skyrwa:hasAssetClass ?assetClass ;
               skyrwa:complianceScore ?compliance ;
               skyrwa:dataQualityScore ?quality ;
               skyrwa:hasRightsProfile ?rp .
        ?rp skyrwa:isTradable "true"^^xsd:boolean .
        FILTER (?compliance >= {min_compliance} && ?quality >= {min_quality})
    }}
    GROUP BY ?assetClass
    HAVING (COUNT(?asset) >= {min_count})
    ORDER BY ?assetClass
    """

    QUERY_CANDIDATES_FOR_CLASS = """
    PREFIX skyrwa: <https://w3id.org/skyrwa#>
    PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

    SELECT ?asset ?flightId
    WHERE {{
        ?asset a skyrwa:AssetCandidate ;
               skyrwa:hasAssetClass "{asset_class}" ;
               skyrwa:flightId ?flightId ;
               skyrwa:complianceScore ?compliance ;
               skyrwa:dataQualityScore ?quality ;
               skyrwa:hasRightsProfile ?rp .
        ?rp skyrwa:isTradable "true"^^xsd:boolean .
        FILTER (?compliance >= {min_compliance} && ?quality >= {min_quality})
    }}
    """

    def find_promotable_groups(self, graph: Graph) -> List[PromotionCandidate]:
        """Identify groups of candidates that meet promotion thresholds."""
        sparql = self.QUERY_PROMOTABLE_BY_CLASS.format(
            min_compliance=self.MIN_COMPLIANCE,
            min_quality=self.MIN_QUALITY,
            min_count=self.MIN_CANDIDATES,
        )
        results: List[PromotionCandidate] = []
        for row in graph.query(sparql):
            asset_class = str(row[0])
            count = int(row[1])
            avg_q = float(row[2])
            avg_c = float(row[3])

            detail_sparql = self.QUERY_CANDIDATES_FOR_CLASS.format(
                asset_class=asset_class,
                min_compliance=self.MIN_COMPLIANCE,
                min_quality=self.MIN_QUALITY,
            )
            candidates = list(graph.query(detail_sparql))
            results.append(PromotionCandidate(
                product_type=asset_class,
                candidate_iris=[str(c[0]) for c in candidates],
                flight_ids=[str(c[1]) for c in candidates],
                eligible=True,
                reason=(
                    f"{count} candidates with avg quality={avg_q:.2f}, "
                    f"avg compliance={avg_c:.2f} (thresholds: "
                    f"count>={self.MIN_CANDIDATES}, "
                    f"quality>={self.MIN_QUALITY}, "
                    f"compliance>={self.MIN_COMPLIANCE})"
                ),
            ))
        return results

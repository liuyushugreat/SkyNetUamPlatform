"""ISWC-level pipeline smoke test.

Exercises the full pipeline including semantic layers:
ingest → evidence → sign → governance → valuation → explanation →
RDF export → SHACL validation → product aggregation → SPARQL query.
"""

from datetime import UTC, datetime

import pytest
from rdflib import Graph

from SkyRwa.ingest import FlightIngestRecord, FlightIngestor
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.provenance.signing import Ed25519Signer
from SkyRwa.rights import GovernanceEngine
from SkyRwa.valuation import RuleBasedValuationEngine
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.rdf.serializer import to_turtle, to_jsonld
from SkyRwa.semantic_rules.validation_runner import ShaclValidator
from SkyRwa.semantic_rules.explanation_rules import ExplanationBuilder
from SkyRwa.semantic_rules.governance_rules import GovernanceRuleEngine
from SkyRwa.productization import CandidateAggregator, ProductBuilder, ProductCatalogue
from SkyRwa.valuation.product_valuation import ProductValuationEngine
from SkyRwa.models.enums import AssetClass


def _make_record(i):
    return FlightIngestRecord(
        flight_id=f"FLT-ISWC-{i:03d}",
        uav_id=f"UAV-I{i%3+1}",
        mission_id=f"MSN-ISWC-{i}",
        operator_id="OP-ISWC",
        start_time=datetime(2026, 4, 1, 8 + i, tzinfo=UTC),
        end_time=datetime(2026, 4, 1, 8 + i, 25, tzinfo=UTC),
        mission_type="route_survey",
        mission_completed=True,
        completion_pct=100.0,
        telemetry_points=1200 + i * 50,
        avg_altitude_m=150.0,
        weather_condition="clear",
        wind_speed_mps=3.0,
        visibility_km=15.0,
    )


class TestISWCPipelineSmoke:
    def test_full_semantic_pipeline(self):
        """End-to-end: 5 flights → evidence → RDF → SHACL → product."""
        ingestor = FlightIngestor()
        eb = EvidenceBuilder()
        signer = Ed25519Signer.generate_keypair("iswc-test")
        gov = GovernanceEngine()
        val = RuleBasedValuationEngine()

        g = Graph()
        bind_namespaces(g)
        mapper = SkyRwaMapper(g)
        units = []

        for i in range(5):
            rec = _make_record(i)
            unit = ingestor.ingest(rec)
            unit = eb.build(unit, rec)
            signer.sign_evidence(unit.evidence)
            gov.govern(unit)
            val.evaluate(unit)
            mapper.map_asset_unit(unit)
            units.append(unit)

        # Graph populated
        assert len(g) > 50

        # Turtle output
        ttl = to_turtle(units[0])
        assert "FlightEvidence" in ttl

        # JSON-LD output
        jld = to_jsonld(units[0])
        assert "FlightEvidence" in jld

        # SHACL validation
        validator = ShaclValidator()
        report = validator.validate(g)
        assert report.conforms, f"SHACL violations: {report.raw_text}"

        # Governance rules
        gov_results = GovernanceRuleEngine.run_all(g)
        assert len(gov_results) >= 3

        # Explanation
        exp = ExplanationBuilder.explain_valuation(units[0])
        assert exp.subject_type == "valuation"

        # Product aggregation
        agg = CandidateAggregator(min_count=3)
        groups = agg.group(units)
        assert len(groups) >= 1

        builder = ProductBuilder()
        prod_val = ProductValuationEngine()
        catalogue = ProductCatalogue()

        for cls, group in groups.items():
            product = builder.build(group)
            pval = prod_val.valuate(product)
            catalogue.register(product)
            assert pval.final_value > 0

        assert len(catalogue) >= 1

        # Catalogue RDF
        cat_graph = catalogue.to_graph()
        assert len(cat_graph) > 0

    def test_governance_failure_not_promotable(self):
        """Flights with violations should not be promotable."""
        rec = FlightIngestRecord(
            flight_id="FLT-ISWC-FAIL",
            uav_id="UAV-FAIL",
            operator_id="OP-FAIL",
            start_time=datetime(2026, 4, 1, 8, tzinfo=UTC),
            end_time=datetime(2026, 4, 1, 8, 15, tzinfo=UTC),
            mission_completed=False,
            completion_pct=30.0,
            violations=["nfz_incursion", "altitude_violation"],
            no_fly_zone_incursions=3,
            telemetry_points=100,
        )
        ingestor = FlightIngestor()
        eb = EvidenceBuilder()
        gov = GovernanceEngine()
        val = RuleBasedValuationEngine()

        unit = ingestor.ingest(rec)
        unit = eb.build(unit, rec)
        gov.govern(unit)
        val.evaluate(unit)

        assert unit.rights_profile is not None
        assert not unit.rights_profile.tradable

        exp = ExplanationBuilder.explain_promotion_eligibility(unit)
        assert "Not eligible" in exp.conclusion

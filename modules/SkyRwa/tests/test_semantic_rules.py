"""Tests for semantic governance, promotion, and explanation rules."""

from datetime import UTC, datetime

import pytest
from rdflib import Graph, Literal
from rdflib.namespace import RDF, XSD

from SkyRwa.rdf.namespaces import SKYRWA, SKYRWA_INST, bind_namespaces
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.models.evidence import FlightEvidencePackage
from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.models.rights import RightsProfile, RevenueParticipant
from SkyRwa.models.valuation import ValuationResultV2
from SkyRwa.models.enums import AssetClass, UsageLevel
from SkyRwa.semantic_rules.governance_rules import GovernanceRuleEngine
from SkyRwa.semantic_rules.explanation_rules import ExplanationBuilder


def _make_unit(flight_id, compliance=0.9, risk=0.1, tradable=True, quality=0.8):
    ev = FlightEvidencePackage(
        flight_id=flight_id, uav_id="UAV-T",
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
        digest_hash="d" * 64,
    )
    return FlightAssetUnit(
        flight_id=flight_id, uav_id="UAV-T",
        evidence=ev,
        compliance_score=compliance,
        risk_score=risk,
        data_quality_score=quality,
        rights_profile=RightsProfile(
            owner="OP", tradable=tradable,
            permitted_uses=[UsageLevel.LICENSED_EXTERNAL],
            revenue_split=[
                RevenueParticipant(party_id="OP", role="operator", share_pct=100.0),
            ],
        ),
        valuation_result=ValuationResultV2(
            asset_unit_id="x", estimated_value=50.0,
            breakdown={"completeness": 0.8, "scarcity": 0.6},
        ),
    )


class TestGovernanceRules:
    def test_low_compliance_flagged(self):
        unit = _make_unit("FLT-GOV-FAIL", compliance=0.3, risk=0.9)
        mapper = SkyRwaMapper()
        mapper.map_asset_unit(unit)
        results = GovernanceRuleEngine.run_all(mapper.graph)
        non_transfer = next(r for r in results if r.rule_id == "GOV-001")
        assert len(non_transfer.affected_assets) >= 1

    def test_clean_asset_not_flagged(self):
        unit = _make_unit("FLT-GOV-OK", compliance=0.95, risk=0.05)
        mapper = SkyRwaMapper()
        mapper.map_asset_unit(unit)
        results = GovernanceRuleEngine.run_all(mapper.graph)
        non_transfer = next(r for r in results if r.rule_id == "GOV-001")
        assert len(non_transfer.affected_assets) == 0

    def test_inject_decisions_adds_triples(self):
        unit = _make_unit("FLT-GOV-INJ", compliance=0.3)
        mapper = SkyRwaMapper()
        mapper.map_asset_unit(unit)
        results = GovernanceRuleEngine.run_all(mapper.graph)
        before = len(mapper.graph)
        GovernanceRuleEngine.inject_decisions(mapper.graph, results)
        assert len(mapper.graph) > before


class TestExplanationBuilder:
    def test_valuation_explanation(self):
        unit = _make_unit("FLT-EXP-VAL")
        exp = ExplanationBuilder.explain_valuation(unit)
        assert exp.subject_type == "valuation"
        assert len(exp.factors) == 2
        assert "50.00" in exp.conclusion

    def test_governance_explanation_tradable(self):
        unit = _make_unit("FLT-EXP-GOV", tradable=True, compliance=0.95)
        exp = ExplanationBuilder.explain_governance(unit)
        assert "Tradable" in exp.conclusion

    def test_governance_explanation_non_tradable(self):
        unit = _make_unit("FLT-EXP-NT", tradable=False)
        exp = ExplanationBuilder.explain_governance(unit)
        assert "Non-transferable" in exp.conclusion

    def test_promotion_eligibility_pass(self):
        unit = _make_unit("FLT-EXP-PROM", compliance=0.9, quality=0.8, tradable=True)
        exp = ExplanationBuilder.explain_promotion_eligibility(unit)
        assert "Eligible" in exp.conclusion

    def test_promotion_eligibility_fail(self):
        unit = _make_unit("FLT-EXP-NOPROM", compliance=0.5, quality=0.3, tradable=False)
        exp = ExplanationBuilder.explain_promotion_eligibility(unit)
        assert "Not eligible" in exp.conclusion

    def test_inject_explanation_to_graph(self):
        unit = _make_unit("FLT-EXP-INJ")
        exp = ExplanationBuilder.explain_valuation(unit)
        g = Graph()
        bind_namespaces(g)
        iri = ExplanationBuilder.inject_explanation(g, exp)
        assert len(g) > 0
        assert (iri, RDF.type, SKYRWA.ValuationExplanation) in g

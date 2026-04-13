"""Tests for RDF mapping of domain objects."""

from datetime import UTC, datetime

import pytest
from rdflib import Graph
from rdflib.namespace import RDF

from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import SKYRWA
from SkyRwa.models.evidence import FlightEvidencePackage
from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.models.rights import RightsProfile, RevenueParticipant
from SkyRwa.models.settlement import RevenueLog, SettlementRecord, SplitEntry
from SkyRwa.models.valuation import ValuationResultV2
from SkyRwa.models.enums import AssetClass, UsageLevel, UsageType


@pytest.fixture
def evidence():
    return FlightEvidencePackage(
        flight_id="FLT-RDF-001",
        uav_id="UAV-R1",
        mission_id="MSN-001",
        operator_id="OP-001",
        start_time=datetime(2026, 3, 1, 10, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 1, 10, 30, tzinfo=UTC),
        digest_hash="abc123",
    )


@pytest.fixture
def asset_unit(evidence):
    return FlightAssetUnit(
        flight_id="FLT-RDF-001",
        uav_id="UAV-R1",
        evidence=evidence,
        compliance_score=0.9,
        data_quality_score=0.8,
        rights_profile=RightsProfile(
            owner="OP-001",
            tradable=True,
            permitted_uses=[UsageLevel.LICENSED_EXTERNAL],
            revenue_split=[
                RevenueParticipant(party_id="OP-001", role="operator", share_pct=50.0),
                RevenueParticipant(party_id="PLATFORM", role="platform", share_pct=50.0),
            ],
        ),
        valuation_result=ValuationResultV2(
            asset_unit_id="test",
            estimated_value=75.0,
            breakdown={"completeness": 0.9, "scarcity": 0.7},
        ),
    )


class TestEvidenceMapping:
    def test_maps_to_flight_evidence_type(self, evidence):
        mapper = SkyRwaMapper()
        iri = mapper.map_evidence(evidence)
        assert (iri, RDF.type, SKYRWA.FlightEvidence) in mapper.graph

    def test_maps_flight_id(self, evidence):
        mapper = SkyRwaMapper()
        mapper.map_evidence(evidence)
        triples = list(mapper.graph.triples((None, SKYRWA.flightId, None)))
        assert any(str(t[2]) == "FLT-RDF-001" for t in triples)

    def test_maps_operator(self, evidence):
        mapper = SkyRwaMapper()
        mapper.map_evidence(evidence)
        ops = list(mapper.graph.triples((None, RDF.type, SKYRWA.Operator)))
        assert len(ops) >= 1

    def test_maps_digest(self, evidence):
        mapper = SkyRwaMapper()
        mapper.map_evidence(evidence)
        digests = list(mapper.graph.triples((None, SKYRWA.hasDigest, None)))
        assert any(str(d[2]) == "abc123" for d in digests)


class TestAssetUnitMapping:
    def test_maps_asset_candidate_type(self, asset_unit):
        mapper = SkyRwaMapper()
        mapper.map_asset_unit(asset_unit)
        assert any(
            t[2] == SKYRWA.AssetCandidate
            for t in mapper.graph.triples((None, RDF.type, None))
        )

    def test_maps_derived_from_evidence(self, asset_unit):
        mapper = SkyRwaMapper()
        mapper.map_asset_unit(asset_unit)
        derivations = list(mapper.graph.triples(
            (None, SKYRWA.derivedFromEvidence, None)
        ))
        assert len(derivations) == 1

    def test_maps_rights_profile(self, asset_unit):
        mapper = SkyRwaMapper()
        mapper.map_asset_unit(asset_unit)
        rights = list(mapper.graph.triples((None, SKYRWA.hasRightsProfile, None)))
        assert len(rights) >= 1

    def test_maps_valuation_factors(self, asset_unit):
        mapper = SkyRwaMapper()
        mapper.map_asset_unit(asset_unit)
        factors = list(mapper.graph.triples(
            (None, SKYRWA.hasValuationFactor, None)
        ))
        assert len(factors) == 2  # completeness + scarcity

    def test_triple_count_reasonable(self, asset_unit):
        mapper = SkyRwaMapper()
        mapper.map_asset_unit(asset_unit)
        assert len(mapper.graph) >= 15


class TestSettlementMapping:
    def test_maps_revenue_log(self):
        log = RevenueLog(
            asset_unit_id="A1",
            usage_type=UsageType.API_CALL,
            consumer="CONSUMER-1",
            gross_amount=100.0,
        )
        mapper = SkyRwaMapper()
        iri = mapper.map_revenue_log(log)
        assert (iri, RDF.type, SKYRWA.UsageEvent) in mapper.graph

    def test_maps_settlement_record(self):
        rec = SettlementRecord(
            asset_unit_id="A1",
            total_gross=100.0,
            participant_totals=[
                SplitEntry(party_id="OP", role="operator", share_pct=60.0, amount=60.0),
            ],
        )
        mapper = SkyRwaMapper()
        iri = mapper.map_settlement_record(rec)
        assert (iri, RDF.type, SKYRWA.SettlementRecord) in mapper.graph

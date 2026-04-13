"""Tests for SHACL shape validation."""

from datetime import UTC, datetime

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from SkyRwa.rdf.namespaces import SKYRWA, SKYRWA_INST, bind_namespaces
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.semantic_rules.validation_runner import ShaclValidator
from SkyRwa.models.evidence import FlightEvidencePackage
from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.models.valuation import ValuationResultV2


@pytest.fixture
def validator():
    return ShaclValidator()


@pytest.fixture
def valid_evidence():
    return FlightEvidencePackage(
        flight_id="FLT-SHACL-001",
        uav_id="UAV-V1",
        start_time=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, 8, 30, tzinfo=UTC),
        digest_hash="sha256valid",
    )


class TestFlightEvidenceShape:
    def test_valid_evidence_conforms(self, validator, valid_evidence):
        mapper = SkyRwaMapper()
        mapper.map_evidence(valid_evidence)
        report = validator.validate(mapper.graph)
        assert report.conforms, f"Expected conformance, got: {report.raw_text}"

    def test_missing_digest_violates(self, validator):
        """Evidence without a digest should trigger a SHACL violation."""
        g = Graph()
        bind_namespaces(g)
        subj = SKYRWA_INST["evidence:no-digest"]
        g.add((subj, RDF.type, SKYRWA.FlightEvidence))
        g.add((subj, SKYRWA.flightId, Literal("FLT-NO-DIGEST")))
        g.add((subj, SKYRWA.uavId, Literal("UAV-X")))
        g.add((subj, SKYRWA.startTime,
               Literal("2026-01-01T08:00:00+00:00", datatype=XSD.dateTime)))
        g.add((subj, SKYRWA.endTime,
               Literal("2026-01-01T08:30:00+00:00", datatype=XSD.dateTime)))
        report = validator.validate(g)
        assert not report.conforms

    def test_missing_flight_id_violates(self, validator):
        g = Graph()
        bind_namespaces(g)
        subj = SKYRWA_INST["evidence:no-fid"]
        g.add((subj, RDF.type, SKYRWA.FlightEvidence))
        g.add((subj, SKYRWA.hasDigest, Literal("abc")))
        g.add((subj, SKYRWA.uavId, Literal("UAV-X")))
        g.add((subj, SKYRWA.startTime,
               Literal("2026-01-01T08:00:00+00:00", datatype=XSD.dateTime)))
        g.add((subj, SKYRWA.endTime,
               Literal("2026-01-01T08:30:00+00:00", datatype=XSD.dateTime)))
        report = validator.validate(g)
        assert not report.conforms


class TestAssetCandidateShape:
    def test_asset_without_evidence_violates(self, validator):
        """An AssetCandidate that lacks derivedFromEvidence should fail."""
        g = Graph()
        bind_namespaces(g)
        subj = SKYRWA_INST["asset:no-evidence"]
        g.add((subj, RDF.type, SKYRWA.AssetCandidate))
        g.add((subj, SKYRWA.hasAssetClass, Literal("flight_evidence")))
        report = validator.validate(g)
        assert not report.conforms


class TestUsageEventShape:
    def test_usage_event_without_consumer_violates(self, validator):
        g = Graph()
        bind_namespaces(g)
        subj = SKYRWA_INST["usage:no-consumer"]
        g.add((subj, RDF.type, SKYRWA.UsageEvent))
        g.add((subj, SKYRWA.grossAmount, Literal(50.0, datatype=XSD.float)))
        report = validator.validate(g)
        assert not report.conforms

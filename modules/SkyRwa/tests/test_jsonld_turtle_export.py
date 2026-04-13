"""Tests for JSON-LD and Turtle serialization."""

from datetime import UTC, datetime

import pytest

from SkyRwa.models.evidence import FlightEvidencePackage
from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.rdf.serializer import to_turtle, to_jsonld, to_graph


@pytest.fixture
def evidence():
    return FlightEvidencePackage(
        flight_id="FLT-SER-001",
        uav_id="UAV-S1",
        start_time=datetime(2026, 4, 1, 8, 0, tzinfo=UTC),
        end_time=datetime(2026, 4, 1, 8, 30, tzinfo=UTC),
        digest_hash="deadbeef",
    )


@pytest.fixture
def asset_unit(evidence):
    return FlightAssetUnit(
        flight_id="FLT-SER-001",
        uav_id="UAV-S1",
        evidence=evidence,
    )


class TestTurtleExport:
    def test_turtle_is_string(self, evidence):
        result = to_turtle(evidence)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_turtle_contains_flight_id(self, evidence):
        result = to_turtle(evidence)
        assert "FLT-SER-001" in result

    def test_turtle_contains_type(self, evidence):
        result = to_turtle(evidence)
        assert "FlightEvidence" in result

    def test_asset_unit_turtle(self, asset_unit):
        result = to_turtle(asset_unit)
        assert "AssetCandidate" in result
        assert "derivedFromEvidence" in result


class TestJsonLdExport:
    def test_jsonld_is_string(self, evidence):
        result = to_jsonld(evidence)
        assert isinstance(result, str)

    def test_jsonld_parseable(self, evidence):
        import json
        result = to_jsonld(evidence)
        data = json.loads(result)
        assert isinstance(data, (dict, list))

    def test_jsonld_contains_type(self, evidence):
        result = to_jsonld(evidence)
        assert "FlightEvidence" in result


class TestGraphExport:
    def test_to_graph_returns_graph(self, evidence):
        g = to_graph(evidence)
        assert len(g) > 0

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            to_graph("not a model")

"""Tests for the provenance / evidence-builder layer."""

from __future__ import annotations

import pytest

from SkyRwa.ingest.flight_ingestor import FlightIngestRecord, FlightIngestor
from SkyRwa.models.enums import AssetStatus
from SkyRwa.provenance.evidence_builder import EvidenceBuilder


class TestEvidenceBuilder:
    def test_build_sets_status(self, ingested_unit, sample_record):
        builder = EvidenceBuilder()
        builder.build(ingested_unit, sample_record)
        assert ingested_unit.status == AssetStatus.EVIDENCE_BUILT

    def test_evidence_has_digest_hash(self, ingested_unit, sample_record):
        builder = EvidenceBuilder()
        builder.build(ingested_unit, sample_record)
        ev = ingested_unit.evidence
        assert ev is not None
        assert len(ev.digest_hash) == 64  # SHA-256 hex

    def test_digest_is_verifiable(self, ingested_unit, sample_record):
        builder = EvidenceBuilder()
        builder.build(ingested_unit, sample_record)
        assert EvidenceBuilder.verify_digest(ingested_unit.evidence)

    def test_tampered_digest_fails(self, ingested_unit, sample_record):
        builder = EvidenceBuilder()
        builder.build(ingested_unit, sample_record)
        ingested_unit.evidence.telemetry_summary.total_points = 999999
        assert not EvidenceBuilder.verify_digest(ingested_unit.evidence)

    def test_signer_fields(self, ingested_unit, sample_record):
        builder = EvidenceBuilder()
        builder.build(ingested_unit, sample_record, signer_id="my-signer")
        ev = ingested_unit.evidence
        assert ev.signed_by == "my-signer"
        assert ev.signature is not None
        assert ev.signed_at is not None

    def test_telemetry_summary_populated(self, ingested_unit, sample_record):
        EvidenceBuilder().build(ingested_unit, sample_record)
        ts = ingested_unit.evidence.telemetry_summary
        assert ts.total_points == sample_record.telemetry_points
        assert ts.avg_altitude_m == sample_record.avg_altitude_m
        assert ts.payload_active == sample_record.payload_active

    def test_environment_populated(self, ingested_unit, sample_record):
        EvidenceBuilder().build(ingested_unit, sample_record)
        env = ingested_unit.evidence.environment
        assert env.weather_condition == sample_record.weather_condition
        assert env.wind_speed_mps == sample_record.wind_speed_mps

    def test_mission_result_populated(self, ingested_unit, sample_record):
        EvidenceBuilder().build(ingested_unit, sample_record)
        mr = ingested_unit.evidence.mission_result
        assert mr.completed == sample_record.mission_completed
        assert mr.deviation_m == sample_record.deviation_m

    def test_empty_flight_id_raises(self, ingested_unit, sample_record):
        sample_record.flight_id = ""
        with pytest.raises(ValueError, match="flight_id"):
            EvidenceBuilder().build(ingested_unit, sample_record)

    def test_digest_is_deterministic(self, ingested_unit, sample_record):
        """Re-computing the digest on the same package always yields the same hash."""
        EvidenceBuilder().build(ingested_unit, sample_record)
        ev = ingested_unit.evidence
        hash1 = EvidenceBuilder._compute_digest(ev)
        hash2 = EvidenceBuilder._compute_digest(ev)
        assert hash1 == hash2 == ev.digest_hash

"""Tests for Ed25519 signature and provenance chain."""

from datetime import UTC, datetime

import pytest

from SkyRwa.models.evidence import FlightEvidencePackage
from SkyRwa.provenance.signing import Ed25519Signer
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.ingest import FlightIngestRecord, FlightIngestor
from SkyRwa.models.asset_unit import FlightAssetUnit


@pytest.fixture
def signer():
    return Ed25519Signer.generate_keypair("test-signer")


@pytest.fixture
def evidence():
    return FlightEvidencePackage(
        flight_id="FLT-SIG-001",
        uav_id="UAV-S1",
        start_time=datetime(2026, 3, 1, 10, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 1, 10, 30, tzinfo=UTC),
        digest_hash="a" * 64,
    )


class TestEd25519Signer:
    def test_sign_and_verify(self, signer, evidence):
        signer.sign_evidence(evidence)
        assert evidence.signature is not None
        assert evidence.signed_by == "test-signer"
        assert evidence.signed_at is not None
        assert signer.verify_evidence(evidence)

    def test_tampered_digest_fails(self, signer, evidence):
        signer.sign_evidence(evidence)
        evidence.digest_hash = "tampered"
        assert not signer.verify_evidence(evidence)

    def test_unsigned_evidence_fails(self, signer, evidence):
        assert not signer.verify_evidence(evidence)

    def test_different_signer_fails(self, signer, evidence):
        signer.sign_evidence(evidence)
        other = Ed25519Signer.generate_keypair("other")
        assert not other.verify_evidence(evidence)

    def test_sign_without_digest_raises(self, signer):
        ev = FlightEvidencePackage(
            flight_id="FLT-NO-DIGEST",
            uav_id="UAV-X",
            start_time=datetime(2026, 1, 1, tzinfo=UTC),
            end_time=datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
        )
        with pytest.raises(ValueError):
            signer.sign_evidence(ev)

    def test_public_key_b64(self, signer):
        b64 = signer.public_key_b64()
        assert isinstance(b64, str)
        assert len(b64) > 10


class TestSignatureChain:
    def test_full_chain_ingest_to_signed_evidence(self, signer):
        """Full lifecycle: ingest → build evidence → sign → verify."""
        record = FlightIngestRecord(
            flight_id="FLT-CHAIN-001",
            uav_id="UAV-CHAIN",
            mission_id="MSN-CHAIN",
            operator_id="OP-CHAIN",
            start_time=datetime(2026, 5, 1, 8, 0, tzinfo=UTC),
            end_time=datetime(2026, 5, 1, 8, 25, tzinfo=UTC),
            mission_completed=True,
            completion_pct=100.0,
            telemetry_points=1000,
        )
        ingestor = FlightIngestor()
        unit = ingestor.ingest(record)

        eb = EvidenceBuilder()
        unit = eb.build(unit, record)

        assert unit.evidence is not None
        assert unit.evidence.digest_hash != ""

        signer.sign_evidence(unit.evidence)
        assert signer.verify_evidence(unit.evidence)

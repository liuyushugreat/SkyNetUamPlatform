"""Shared test fixtures for SkyRwa tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from SkyRwa.ingest.flight_ingestor import FlightIngestRecord, FlightIngestor
from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.models.enums import AssetStatus, UsageType
from SkyRwa.models.settlement import SettlementRule, SplitEntry
from SkyRwa.provenance.evidence_builder import EvidenceBuilder
from SkyRwa.rights.governance import GovernanceEngine


@pytest.fixture()
def sample_record() -> FlightIngestRecord:
    now = datetime.now(UTC)
    return FlightIngestRecord(
        flight_id="FLT-TEST-001",
        uav_id="UAV-TEST-01",
        mission_id="MSN-TEST-01",
        operator_id="OP-TEST",
        mission_type="route_survey",
        start_time=now - timedelta(hours=1),
        end_time=now,
        telemetry_points=3600,
        avg_altitude_m=120.0,
        max_altitude_m=150.0,
        max_speed_mps=15.0,
        avg_speed_mps=10.0,
        min_battery_pct=30.0,
        avg_battery_pct=65.0,
        payload_active=True,
        trajectory_length_m=5000.0,
        weather_condition="clear",
        wind_speed_mps=3.0,
        visibility_km=10.0,
        temperature_c=25.0,
        mission_completed=True,
        completion_pct=100.0,
        deviation_m=5.0,
        raw_data_uri="file:///tmp/test-flight.parquet",
        raw_data_hash="sha256:aabbccdd",
        trajectory_hash="sha256:11223344",
    )


@pytest.fixture()
def ingested_unit(sample_record: FlightIngestRecord) -> FlightAssetUnit:
    return FlightIngestor().ingest(sample_record)


@pytest.fixture()
def evidence_unit(
    ingested_unit: FlightAssetUnit,
    sample_record: FlightIngestRecord,
) -> FlightAssetUnit:
    EvidenceBuilder().build(ingested_unit, sample_record, signer_id="test-signer")
    return ingested_unit


@pytest.fixture()
def governed_unit(evidence_unit: FlightAssetUnit) -> FlightAssetUnit:
    GovernanceEngine().govern(evidence_unit, owner="test-owner", operator_id="OP-TEST")
    return evidence_unit


@pytest.fixture()
def default_settlement_rule() -> SettlementRule:
    return SettlementRule(
        trigger_types=[UsageType.API_CALL],
        participants=[
            SplitEntry(party_id="platform", role="platform", share_pct=30),
            SplitEntry(party_id="operator", role="operator", share_pct=50),
            SplitEntry(party_id="processor", role="data_processor", share_pct=20),
        ],
        min_settlement_unit=0.01,
    )

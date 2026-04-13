"""Smoke tests for the full flight-to-asset pipeline."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta

import pytest

from SkyRwa.ingest.flight_ingestor import FlightIngestRecord
from SkyRwa.models.enums import AssetClass, AssetStatus, UsageType
from SkyRwa.models.settlement import SettlementRule, SplitEntry
from SkyRwa.pipeline.flight_to_asset import FlightToAssetPipeline
from SkyRwa.provenance.evidence_builder import EvidenceBuilder
from SkyRwa.settlement.ledger import Ledger
from SkyRwa.storage.json_store import JsonStore


def _make_record(**overrides) -> FlightIngestRecord:
    now = datetime.now(UTC)
    defaults = dict(
        flight_id="FLT-SMOKE-001",
        uav_id="UAV-SMOKE",
        mission_id="MSN-SMOKE",
        operator_id="OP-SMOKE",
        mission_type="route_survey",
        start_time=now - timedelta(hours=1),
        end_time=now,
        telemetry_points=3600,
        avg_altitude_m=100.0,
        max_altitude_m=130.0,
        max_speed_mps=12.0,
        avg_speed_mps=8.0,
        min_battery_pct=25.0,
        avg_battery_pct=60.0,
        trajectory_length_m=4000.0,
        mission_completed=True,
        completion_pct=100.0,
        raw_data_hash="sha256:smoke",
        trajectory_hash="sha256:smoke-traj",
    )
    defaults.update(overrides)
    return FlightIngestRecord(**defaults)


class TestPipelineSmoke:
    def test_pipeline_produces_valuated_unit(self):
        pipeline = FlightToAssetPipeline()
        record = _make_record()
        unit = pipeline.run(record)
        assert unit.status == AssetStatus.VALUATED
        assert unit.valuation_result is not None
        assert unit.evidence is not None
        assert unit.rights_profile is not None

    def test_pipeline_with_settlement_rule(self):
        rule = SettlementRule(
            participants=[
                SplitEntry(party_id="p", role="platform", share_pct=40),
                SplitEntry(party_id="o", role="operator", share_pct=60),
            ],
        )
        pipeline = FlightToAssetPipeline(default_settlement_rule=rule)
        unit = pipeline.run(_make_record())
        assert unit.settlement_rule is not None
        assert len(unit.settlement_rule.participants) == 2

    def test_pipeline_with_violations_limits_rights(self):
        record = _make_record(violations=["airspace_breach"])
        pipeline = FlightToAssetPipeline()
        unit = pipeline.run(record)
        rp = unit.rights_profile
        assert rp is not None
        assert not rp.tradable

    def test_evidence_digest_verifiable_after_pipeline(self):
        pipeline = FlightToAssetPipeline(signer_id="smoke-signer")
        unit = pipeline.run(_make_record())
        assert EvidenceBuilder.verify_digest(unit.evidence)

    def test_json_store_roundtrip(self):
        pipeline = FlightToAssetPipeline()
        unit = pipeline.run(_make_record())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonStore(base_dir=tmpdir)
            store.save(unit)
            reloaded = store.load(unit.asset_unit_id)
            assert reloaded is not None
            assert reloaded.flight_id == unit.flight_id
            assert reloaded.valuation_result is not None

    def test_ledger_after_pipeline(self):
        rule = SettlementRule(
            participants=[
                SplitEntry(party_id="p", role="platform", share_pct=50),
                SplitEntry(party_id="o", role="operator", share_pct=50),
            ],
        )
        pipeline = FlightToAssetPipeline(default_settlement_rule=rule)
        unit = pipeline.run(_make_record())
        ledger = Ledger()
        log = ledger.record_usage(unit, UsageType.API_CALL, "consumer-x", 20.0)
        assert len(log.split_detail) == 2
        assert abs(sum(s.amount for s in log.split_detail) - 20.0) < 0.01

    def test_clean_survey_gets_route_optimization_class(self):
        record = _make_record(mission_type="route_survey")
        pipeline = FlightToAssetPipeline()
        unit = pipeline.run(record)
        assert unit.asset_class in (
            AssetClass.ROUTE_OPTIMIZATION_SAMPLE,
            AssetClass.FLIGHT_EVIDENCE,
        )

    def test_empty_flight_id_raises(self):
        record = _make_record(flight_id="")
        pipeline = FlightToAssetPipeline()
        with pytest.raises(ValueError, match="flight_id"):
            pipeline.run(record)

    def test_full_9_step_workflow(self):
        """End-to-end: ingest -> evidence -> govern -> valuate -> settle."""
        rule = SettlementRule(
            participants=[
                SplitEntry(party_id="p", role="platform", share_pct=30),
                SplitEntry(party_id="o", role="operator", share_pct=50),
                SplitEntry(party_id="d", role="data_processor", share_pct=20),
            ],
        )
        pipeline = FlightToAssetPipeline(
            default_settlement_rule=rule,
            signer_id="test-signer",
        )
        unit = pipeline.run(_make_record())
        assert unit.status == AssetStatus.VALUATED

        ledger = Ledger()
        ledger.record_usage(unit, UsageType.API_CALL, "consumer-A", 10.0)
        ledger.record_usage(unit, UsageType.TRAINING_USE, "consumer-B", 5.0)

        record = ledger.settle_all(unit.asset_unit_id)
        assert record is not None
        assert record.total_gross == 15.0
        assert len(record.participant_totals) == 3

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonStore(base_dir=tmpdir)
            store.save(unit)
            store.save_ledger(ledger.to_dicts())
            store.save_settlements(ledger.settlements_to_dicts())
            assert len(store.load_settlements()) == 1
            assert len(store.load_ledger()) == 2

#!/usr/bin/env python3
"""
Demo: end-to-end flight -> data-asset-candidate pipeline
==========================================================

Walks through **all 9 steps** of the minimum working workflow:

1. Input a simulated flight record
2. Generate FlightEvidencePackage (with SHA-256 digest)
3. Generate RightsProfile (governance)
4. Run valuation logic -> ValuationResultV2
5. Generate FlightAssetUnit
6. Persist asset unit + ledger + settlements as JSON
7. Simulate a "data consumed" revenue event
8. Auto-split revenue via SettlementRule
9. Output final settlement record

Run from the repository root::

    python -m SkyRwa.examples.demo_flight_to_asset

or directly::

    python modules/SkyRwa/examples/demo_flight_to_asset.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODULE_ROOT = _HERE.parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from SkyRwa.ingest.flight_ingestor import FlightIngestRecord
from SkyRwa.models.enums import UsageType
from SkyRwa.models.settlement import SettlementRule, SplitEntry
from SkyRwa.pipeline.flight_to_asset import FlightToAssetPipeline
from SkyRwa.provenance.evidence_builder import EvidenceBuilder
from SkyRwa.settlement.ledger import Ledger
from SkyRwa.storage.json_store import JsonStore

SEP = "-" * 68


def make_sample_record() -> FlightIngestRecord:
    """Build a realistic-looking flight ingest record."""
    now = datetime.now(UTC)
    return FlightIngestRecord(
        flight_id="FLT-20260413-0042",
        uav_id="UAV-SZ-007",
        mission_id="MSN-ROUTE-SURVEY-88",
        operator_id="OP-SHENZHEN-12",
        mission_type="route_survey",
        start_time=now - timedelta(hours=1, minutes=23),
        end_time=now,
        waypoints=[
            {"lat": 22.5431, "lon": 114.0579, "alt": 120.0},
            {"lat": 22.5500, "lon": 114.0650, "alt": 150.0},
            {"lat": 22.5580, "lon": 114.0700, "alt": 130.0},
        ],
        corridor_id="COR-SZ-EAST-03",
        telemetry_points=4980,
        avg_altitude_m=133.0,
        max_altitude_m=155.0,
        max_speed_mps=18.5,
        avg_speed_mps=12.3,
        min_battery_pct=22.0,
        avg_battery_pct=61.0,
        payload_active=True,
        trajectory_length_m=8742.0,
        weather_condition="partly_cloudy",
        wind_speed_mps=5.2,
        visibility_km=8.0,
        temperature_c=28.0,
        no_fly_zone_incursions=0,
        risk_events=["mild_turbulence_at_wp2"],
        mission_completed=True,
        completion_pct=100.0,
        deviation_m=12.5,
        raw_data_uri="s3://skyflow-data/flights/FLT-20260413-0042.parquet",
        raw_data_hash="sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        trajectory_hash="sha256:fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
    )


def main() -> None:
    print("=" * 68)
    print("  SkyRwa -- Flight-to-Asset Pipeline Demo (9-step workflow)")
    print("=" * 68)

    # ==================================================================
    # STEP 1: Input a simulated flight record
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 1: Input simulated flight record")
    print(SEP)
    record = make_sample_record()
    print(f"  flight_id   : {record.flight_id}")
    print(f"  uav_id      : {record.uav_id}")
    print(f"  mission     : {record.mission_type}")
    print(f"  operator    : {record.operator_id}")
    print(f"  duration    : {record.start_time} -> {record.end_time}")
    print(f"  telemetry   : {record.telemetry_points} points")

    # --- Configure pipeline ---
    settlement_rule = SettlementRule(
        trigger_types=[UsageType.API_CALL, UsageType.TRAINING_USE],
        participants=[
            SplitEntry(party_id="platform", role="platform", share_pct=30),
            SplitEntry(party_id="OP-SHENZHEN-12", role="operator", share_pct=50),
            SplitEntry(party_id="data-team", role="data_processor", share_pct=20),
        ],
        min_settlement_unit=0.01,
        settlement_cycle_days=30,
    )

    pipeline = FlightToAssetPipeline(
        default_settlement_rule=settlement_rule,
        signer_id="skyrwa-demo-signer",
    )

    # The pipeline runs steps 2-5 internally; we print each result.
    unit = pipeline.run(record, owner="SkyNet-Platform")

    # ==================================================================
    # STEP 2: FlightEvidencePackage generated
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 2: FlightEvidencePackage (provenance)")
    print(SEP)
    ev = unit.evidence
    assert ev is not None, "Evidence must be attached"
    print(f"  evidence_id   : {ev.evidence_id}")
    print(f"  digest_hash   : {ev.digest_hash[:32]}...")
    print(f"  raw_data_hash : {ev.raw_data_hash[:40]}...")
    print(f"  signed_by     : {ev.signed_by}")
    print(f"  duration_sec  : {ev.duration_seconds:.0f}")
    verified = EvidenceBuilder.verify_digest(ev)
    print(f"  digest_valid  : {verified}")

    # ==================================================================
    # STEP 3: RightsProfile generated (governance)
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 3: RightsProfile (governance)")
    print(SEP)
    rp = unit.rights_profile
    assert rp is not None, "RightsProfile must be attached"
    print(f"  owner           : {rp.owner}")
    print(f"  tradable        : {rp.tradable}")
    print(f"  desens_required : {rp.desensitization_required}")
    print(f"  permitted_uses  : {[u.value for u in rp.permitted_uses]}")
    print(f"  data_categories : {[c.value for c in rp.data_categories]}")
    print(f"  revenue_split   :")
    for p in rp.revenue_split:
        print(f"    {p.party_id:20s}  {p.role:16s}  {p.share_pct:.0f}%")

    # ==================================================================
    # STEP 4: Valuation result
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 4: ValuationResultV2 (rule-based engine)")
    print(SEP)
    vr = unit.valuation_result
    assert vr is not None, "Valuation result must be attached"
    print(f"  engine_id   : {vr.engine_id}")
    print(f"  est. value  : {vr.estimated_value:.4f} {vr.currency}")
    print(f"  confidence  : {vr.confidence:.4f}")
    print(f"  quality     :")
    qs = vr.quality_score
    print(f"    completeness        : {qs.completeness:.4f}  (weight 0.25)")
    print(f"    temporal_continuity : {qs.temporal_continuity:.4f}  (weight 0.20)")
    print(f"    sensor_reliability  : {qs.sensor_reliability:.4f}  (weight 0.20)")
    print(f"    event_richness      : {qs.event_richness:.4f}  (weight 0.15)")
    print(f"    compliance_degree   : {qs.compliance_degree:.4f}  (weight 0.20)")
    print(f"    overall             : {qs.overall:.4f}")
    print(f"  value       :")
    vs = vr.value_score
    print(f"    scarcity            : {vs.scarcity:.4f}  (weight 0.25)")
    print(f"    scenario_relevance  : {vs.scenario_relevance:.4f}  (weight 0.25)")
    print(f"    reuse_potential     : {vs.reuse_potential:.4f}  (weight 0.30)")
    print(f"    timeliness          : {vs.timeliness:.4f}  (weight 0.20)")
    print(f"    overall             : {vs.overall:.4f}")
    print(f"  formula     : {vr.notes}")

    # ==================================================================
    # STEP 5: FlightAssetUnit summary
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 5: FlightAssetUnit (aggregate)")
    print(SEP)
    print(f"  asset_unit_id : {unit.asset_unit_id}")
    print(f"  flight_id     : {unit.flight_id}")
    print(f"  status        : {unit.status.value}")
    print(f"  asset_class   : {unit.asset_class.value}")
    print(f"  risk_score    : {unit.risk_score:.4f}")
    print(f"  compliance    : {unit.compliance_score:.4f}")
    print(f"  data_quality  : {unit.data_quality_score:.4f}")

    # ==================================================================
    # STEP 6: Persist to JSON
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 6: Persist to JSON")
    print(SEP)
    tmpdir = tempfile.mkdtemp(prefix="skyrwa_demo_")
    store = JsonStore(base_dir=tmpdir)
    asset_path = store.save(unit)
    print(f"  asset unit  -> {asset_path}")

    # ==================================================================
    # STEP 7: Simulate revenue events (data consumed)
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 7: Simulate revenue events")
    print(SEP)
    ledger = Ledger()

    log1 = ledger.record_usage(
        unit,
        usage_type=UsageType.API_CALL,
        consumer="analytics-team-A",
        gross_amount=5.00,
    )
    print(f"  Event 1: API_CALL by analytics-team-A, gross={log1.gross_amount:.2f}")

    log2 = ledger.record_usage(
        unit,
        usage_type=UsageType.TRAINING_USE,
        consumer="ml-pipeline-B",
        gross_amount=12.50,
    )
    print(f"  Event 2: TRAINING_USE by ml-pipeline-B, gross={log2.gross_amount:.2f}")
    print(f"  Total ledger revenue: {ledger.total_revenue():.2f}")

    # ==================================================================
    # STEP 8: Auto-split revenue per SettlementRule
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 8: Revenue auto-split (from SettlementRule)")
    print(SEP)
    for i, log in enumerate([log1, log2], 1):
        print(f"  Event {i} (gross={log.gross_amount:.2f}):")
        for s in log.split_detail:
            print(f"    {s.party_id:20s}  {s.share_pct:6.2f}%  -> {s.amount:.4f}")

    # ==================================================================
    # STEP 9: Settle & output final settlement record
    # ==================================================================
    print(f"\n{SEP}")
    print("STEP 9: Final settlement record")
    print(SEP)
    settlement = ledger.settle_all(unit.asset_unit_id)
    assert settlement is not None, "Settlement must succeed"
    print(f"  settlement_id     : {settlement.settlement_id}")
    print(f"  asset_unit_id     : {settlement.asset_unit_id}")
    print(f"  settled_usage_ids : {settlement.settled_usage_ids}")
    print(f"  total_gross       : {settlement.total_gross:.4f}")
    print(f"  settled_at        : {settlement.settled_at}")
    print(f"  participant totals:")
    for p in settlement.participant_totals:
        print(f"    {p.party_id:20s}  {p.role:16s}  {p.amount:.4f}")

    # Persist ledger + settlements
    ledger_path = store.save_ledger(ledger.to_dicts())
    settle_path = store.save_settlements(ledger.settlements_to_dicts())
    # Re-save the asset unit (now with revenue_log populated)
    store.save(unit)

    print(f"\n  Persisted files:")
    print(f"    asset unit   -> {asset_path}")
    print(f"    ledger       -> {ledger_path}")
    print(f"    settlements  -> {settle_path}")

    # Verify round-trip
    reloaded = store.load(unit.asset_unit_id)
    assert reloaded is not None
    assert reloaded.flight_id == unit.flight_id
    print(f"\n  Round-trip verification: OK (reloaded {reloaded.asset_unit_id})")

    print(f"\n{'=' * 68}")
    print("  Demo complete.  All 9 steps executed successfully.")
    print("=" * 68)


if __name__ == "__main__":
    main()

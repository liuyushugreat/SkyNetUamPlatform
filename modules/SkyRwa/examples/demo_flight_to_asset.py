#!/usr/bin/env python3
"""
Demo: end-to-end flight → asset-candidate pipeline
====================================================

Run from the repository root::

    python -m SkyRwa.examples.demo_flight_to_asset

or directly::

    python modules/SkyRwa/examples/demo_flight_to_asset.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Ensure the parent package is importable when running as a script.
_HERE = Path(__file__).resolve().parent
_MODULE_ROOT = _HERE.parent.parent  # …/modules
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from SkyRwa.ingest.flight_ingestor import FlightIngestRecord
from SkyRwa.models.enums import UsageType
from SkyRwa.models.settlement import SettlementRule, SplitEntry
from SkyRwa.pipeline.flight_to_asset import FlightToAssetPipeline
from SkyRwa.settlement.ledger import Ledger
from SkyRwa.storage.json_store import JsonStore


def make_sample_record() -> FlightIngestRecord:
    """Build a realistic-looking flight ingest record."""
    now = datetime.utcnow()
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
    print("=" * 70)
    print("  SkyRwa — Flight-to-Asset Pipeline Demo")
    print("=" * 70)

    # --- 1. Configure pipeline ---
    default_rule = SettlementRule(
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
        default_settlement_rule=default_rule,
        signer_id="skyrwa-demo-signer",
    )

    # --- 2. Run pipeline ---
    record = make_sample_record()
    unit = pipeline.run(record, owner="SkyNet-Platform")

    print(f"\n[OK] Asset unit created: {unit.asset_unit_id}")
    print(f"  Flight      : {unit.flight_id}")
    print(f"  Status      : {unit.status.value}")
    print(f"  Asset class : {unit.asset_class.value}")
    print(f"  Risk score  : {unit.risk_score:.4f}")
    print(f"  Compliance  : {unit.compliance_score:.4f}")
    print(f"  Quality     : {unit.data_quality_score:.4f}")

    if unit.valuation_result:
        vr = unit.valuation_result
        print(f"  Est. value  : {vr.estimated_value:.4f} {vr.currency}")
        print(f"  Confidence  : {vr.confidence:.4f}")

    if unit.rights_profile:
        rp = unit.rights_profile
        print(f"  Owner       : {rp.owner}")
        print(f"  Tradable    : {rp.tradable}")
        print(f"  Permitted   : {[u.value for u in rp.permitted_uses]}")

    # --- 3. Simulate revenue events ---
    print("\n--- Revenue simulation ---")
    ledger = Ledger()

    log1 = ledger.record_usage(
        unit,
        usage_type=UsageType.API_CALL,
        consumer="analytics-team-A",
        gross_amount=5.00,
    )
    print(f"  Usage #{log1.usage_id[:8]}  gross={log1.gross_amount}  splits={len(log1.split_detail)}")
    for s in log1.split_detail:
        print(f"    {s.party_id:20s}  {s.share_pct:6.2f}%  → {s.amount:.4f}")

    log2 = ledger.record_usage(
        unit,
        usage_type=UsageType.TRAINING_USE,
        consumer="ml-pipeline-B",
        gross_amount=12.50,
    )
    print(f"  Usage #{log2.usage_id[:8]}  gross={log2.gross_amount}  splits={len(log2.split_detail)}")

    print(f"\n  Total ledger revenue: {ledger.total_revenue():.4f}")

    # --- 4. Persist to temp dir ---
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonStore(base_dir=tmpdir)
        path = store.save(unit)
        print(f"\n[OK] Saved asset unit to: {path}")

        ledger_path = store.save_ledger(ledger.to_dicts())
        print(f"[OK] Saved ledger to: {ledger_path}")

        reloaded = store.load(unit.asset_unit_id)
        assert reloaded is not None
        assert reloaded.flight_id == unit.flight_id
        print(f"[OK] Reloaded and verified: {reloaded.asset_unit_id}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

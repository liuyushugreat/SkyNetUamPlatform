"""Generate benchmark data for ISWC evaluation.

Produces 30+ flights across 8 scenario categories, runs them through
the full pipeline, exports JSON inputs, RDF graphs, and expected labels.

Usage::

    python -m SkyRwa.benchmarks.generate_benchmark --output-dir ./benchmark_output
"""

from __future__ import annotations

import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

# Ensure the package is importable when run as a script
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.models.enums import AssetClass
from SkyRwa.ingest import FlightIngestRecord, FlightIngestor
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.provenance.signing import Ed25519Signer
from SkyRwa.rights import GovernanceEngine
from SkyRwa.valuation import RuleBasedValuationEngine
from SkyRwa.settlement import Ledger, RevenueSplitter
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.models.asset_unit import FlightAssetUnit

from rdflib import Graph

# ── Scenario templates ──────────────────────────────────────────────────────

SCENARIOS: List[dict] = [
    # 1. Clean route survey (×5)
    *[{
        "tag": "clean_route_survey",
        "flight_id": f"FLT-CLEAN-{i:03d}",
        "uav_id": f"UAV-A{i%3+1}",
        "mission_type": "route_survey",
        "asset_class": AssetClass.ROUTE_OPTIMIZATION_SAMPLE,
        "completed": True, "completion_pct": 100.0,
        "violations": [], "anomalies": [],
        "no_fly_zone_incursions": 0, "risk_events": [],
        "weather": "clear", "wind": 3.0, "visibility": 15.0,
        "total_points": 1200 + i * 50,
        "duration_min": 25 + i * 2,
    } for i in range(5)],

    # 2. Night flight (×4)
    *[{
        "tag": "night_flight",
        "flight_id": f"FLT-NIGHT-{i:03d}",
        "uav_id": f"UAV-B{i%2+1}",
        "mission_type": "inspection",
        "asset_class": AssetClass.MAINTENANCE_SAMPLE,
        "completed": True, "completion_pct": 95.0,
        "violations": [], "anomalies": ["low_visibility_warning"],
        "no_fly_zone_incursions": 0, "risk_events": ["nighttime_operation"],
        "weather": "clear_night", "wind": 2.0, "visibility": 5.0,
        "total_points": 800 + i * 30,
        "duration_min": 20,
    } for i in range(4)],

    # 3. Weather disturbance (×4)
    *[{
        "tag": "weather_disturbance",
        "flight_id": f"FLT-WEATHER-{i:03d}",
        "uav_id": f"UAV-C{i%2+1}",
        "mission_type": "delivery",
        "asset_class": AssetClass.WEATHER_OPERATION_SAMPLE,
        "completed": True, "completion_pct": 90.0 - i * 5,
        "violations": [], "anomalies": ["turbulence", "gust_warning"],
        "no_fly_zone_incursions": 0,
        "risk_events": ["weather_degradation", "wind_shear"],
        "weather": "rainy", "wind": 12.0 + i, "visibility": 3.0,
        "total_points": 600 + i * 20,
        "duration_min": 30,
    } for i in range(4)],

    # 4. Near-NFZ event (×3)
    *[{
        "tag": "near_nfz",
        "flight_id": f"FLT-NFZ-{i:03d}",
        "uav_id": f"UAV-D1",
        "mission_type": "patrol",
        "asset_class": AssetClass.COMPLIANCE_RECORD,
        "completed": True, "completion_pct": 100.0,
        "violations": ["nfz_proximity_warning"] if i > 0 else [],
        "anomalies": ["nfz_proximity"],
        "no_fly_zone_incursions": i,
        "risk_events": ["nfz_buffer_entry"],
        "weather": "overcast", "wind": 5.0, "visibility": 8.0,
        "total_points": 1000,
        "duration_min": 35,
    } for i in range(3)],

    # 5. Anomaly-rich maintenance (×4)
    *[{
        "tag": "anomaly_maintenance",
        "flight_id": f"FLT-MAINT-{i:03d}",
        "uav_id": f"UAV-E{i%2+1}",
        "mission_type": "maintenance_check",
        "asset_class": AssetClass.MAINTENANCE_SAMPLE,
        "completed": True, "completion_pct": 85.0,
        "violations": [],
        "anomalies": ["motor_vibration", "battery_temp_high", "gps_drift"][: i + 1],
        "no_fly_zone_incursions": 0,
        "risk_events": ["equipment_warning"],
        "weather": "clear", "wind": 4.0, "visibility": 12.0,
        "total_points": 900,
        "duration_min": 15,
    } for i in range(4)],

    # 6. Emergency logistics (×3)
    *[{
        "tag": "emergency_logistics",
        "flight_id": f"FLT-EMER-{i:03d}",
        "uav_id": f"UAV-F1",
        "mission_type": "emergency_delivery",
        "asset_class": AssetClass.RISK_DATASET,
        "completed": i < 2, "completion_pct": 100.0 if i < 2 else 60.0,
        "violations": ["altitude_exceedance"] if i == 2 else [],
        "anomalies": [] if i < 2 else ["mission_abort"],
        "no_fly_zone_incursions": 0,
        "risk_events": ["priority_corridor"],
        "weather": "clear", "wind": 6.0, "visibility": 10.0,
        "total_points": 500,
        "duration_min": 10,
    } for i in range(3)],

    # 7. Incomplete / low-quality (×4)
    *[{
        "tag": "low_quality",
        "flight_id": f"FLT-LOWQ-{i:03d}",
        "uav_id": f"UAV-G1",
        "mission_type": "survey",
        "asset_class": AssetClass.FLIGHT_EVIDENCE,
        "completed": False, "completion_pct": 30.0 + i * 10,
        "violations": ["data_gap", "sensor_failure"],
        "anomalies": ["telemetry_loss", "gps_failure"],
        "no_fly_zone_incursions": 0,
        "risk_events": [],
        "weather": "foggy", "wind": 8.0, "visibility": 1.0,
        "total_points": 100 + i * 30,
        "duration_min": 5,
    } for i in range(4)],

    # 8. Rights-conflicted aggregation (×3)
    *[{
        "tag": "rights_conflict",
        "flight_id": f"FLT-RIGHTS-{i:03d}",
        "uav_id": f"UAV-H{i+1}",
        "mission_type": "commercial_survey",
        "asset_class": AssetClass.AUDIT_READY_PACKAGE,
        "completed": True, "completion_pct": 98.0,
        "violations": [],
        "anomalies": [],
        "no_fly_zone_incursions": 0,
        "risk_events": [],
        "weather": "clear", "wind": 3.0, "visibility": 15.0,
        "total_points": 1500,
        "duration_min": 40,
    } for i in range(3)],
]


def _build_record(s: dict, base_time: datetime) -> FlightIngestRecord:
    start = base_time + timedelta(hours=random.randint(0, 200))
    end = start + timedelta(minutes=s["duration_min"])
    return FlightIngestRecord(
        flight_id=s["flight_id"],
        uav_id=s["uav_id"],
        mission_id=f"MSN-{s['flight_id']}",
        operator_id="OP-BENCH",
        start_time=start,
        end_time=end,
        mission_type=s["mission_type"],
        telemetry_points=s["total_points"],
        avg_altitude_m=120.0,
        max_altitude_m=200.0,
        max_speed_mps=25.0,
        avg_speed_mps=15.0,
        min_battery_pct=40.0,
        weather_condition=s["weather"],
        wind_speed_mps=s["wind"],
        visibility_km=s["visibility"],
        no_fly_zone_incursions=s["no_fly_zone_incursions"],
        risk_events=s.get("risk_events", []),
        mission_completed=s["completed"],
        completion_pct=s["completion_pct"],
        anomalies=s.get("anomalies", []),
        violations=s.get("violations", []),
    )


def generate(output_dir: Path | None = None) -> dict:
    """Generate benchmark data and return summary statistics."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "sample_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = output_dir.parent / "sample_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2026, 1, 15, 8, 0, 0, tzinfo=UTC)
    ingestor = FlightIngestor()
    evidence_builder = EvidenceBuilder()
    governance = GovernanceEngine()
    valuation = RuleBasedValuationEngine()
    signer = Ed25519Signer.generate_keypair("benchmark-signer")

    g = Graph()
    bind_namespaces(g)
    mapper = SkyRwaMapper(g)

    units: List[FlightAssetUnit] = []
    labels: List[dict] = []

    for scenario in SCENARIOS:
        record = _build_record(scenario, base_time)
        unit = ingestor.ingest(record)
        unit = evidence_builder.build(unit, record)
        signer.sign_evidence(unit.evidence)
        governance.govern(unit)
        valuation.evaluate(unit)

        mapper.map_asset_unit(unit)
        units.append(unit)

        is_tradable = unit.rights_profile.tradable if unit.rights_profile else False
        labels.append({
            "flight_id": scenario["flight_id"],
            "tag": scenario["tag"],
            "asset_class": scenario["asset_class"].value if isinstance(scenario["asset_class"], AssetClass) else scenario["asset_class"],
            "expected_tradable": is_tradable,
            "expected_quality_above_05": unit.data_quality_score >= 0.5,
            "expected_compliance_above_07": unit.compliance_score >= 0.7,
            "has_violations": len(scenario.get("violations", [])) > 0,
            "completed": scenario["completed"],
        })

    # Save JSON inputs
    records_json = [
        {
            "flight_id": s["flight_id"],
            "tag": s["tag"],
            **{k: v for k, v in s.items() if k not in ("tag",)},
            "asset_class": s["asset_class"].value if isinstance(s["asset_class"], AssetClass) else s["asset_class"],
        }
        for s in SCENARIOS
    ]
    (output_dir / "benchmark_flights.json").write_text(
        json.dumps(records_json, indent=2, default=str), encoding="utf-8"
    )

    # Save asset units
    asset_dicts = [u.model_dump(mode="json") for u in units]
    (output_dir / "benchmark_assets.json").write_text(
        json.dumps(asset_dicts, indent=2, default=str), encoding="utf-8"
    )

    # Save labels
    (output_dir / "benchmark_labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8"
    )

    # Save RDF graph
    g.serialize(destination=str(graph_dir / "benchmark_graph.ttl"), format="turtle")
    g.serialize(destination=str(graph_dir / "benchmark_graph.jsonld"), format="json-ld")

    summary = {
        "total_flights": len(SCENARIOS),
        "scenario_tags": list({s["tag"] for s in SCENARIOS}),
        "tradable_count": sum(1 for l in labels if l["expected_tradable"]),
        "non_tradable_count": sum(1 for l in labels if not l["expected_tradable"]),
        "with_violations": sum(1 for l in labels if l["has_violations"]),
        "graph_triples": len(g),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate SkyRwa benchmark data")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = generate(args.output_dir)
    print(f"Generated benchmark: {result['total_flights']} flights, "
          f"{result['graph_triples']} triples")
    for k, v in result.items():
        print(f"  {k}: {v}")

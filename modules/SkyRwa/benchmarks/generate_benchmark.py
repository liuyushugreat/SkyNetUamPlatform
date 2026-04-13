"""Generate benchmark data for ISWC evaluation.

Produces 100+ flights across 10 scenario categories, runs them through
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
from typing import List

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.models.enums import AssetClass
from SkyRwa.ingest import FlightIngestRecord, FlightIngestor
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.provenance.signing import Ed25519Signer
from SkyRwa.rights import GovernanceEngine
from SkyRwa.valuation import RuleBasedValuationEngine
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.models.asset_unit import FlightAssetUnit

from rdflib import Graph

# ── Scenario templates ──────────────────────────────────────────────────────

def _make_scenarios() -> List[dict]:
    scenarios: List[dict] = []

    # 1. Clean route survey (×12)
    for i in range(12):
        scenarios.append({
            "tag": "clean_route_survey",
            "flight_id": f"FLT-CLEAN-{i:03d}",
            "uav_id": f"UAV-A{i % 4 + 1}",
            "mission_type": "route_survey",
            "asset_class": AssetClass.ROUTE_OPTIMIZATION_SAMPLE,
            "completed": True, "completion_pct": 100.0,
            "violations": [], "anomalies": [],
            "no_fly_zone_incursions": 0, "risk_events": [],
            "weather": "clear", "wind": 2.0 + i * 0.3, "visibility": 15.0,
            "total_points": 1000 + i * 80,
            "duration_min": 20 + i * 2,
        })

    # 2. Night flight (×8)
    for i in range(8):
        scenarios.append({
            "tag": "night_flight",
            "flight_id": f"FLT-NIGHT-{i:03d}",
            "uav_id": f"UAV-B{i % 3 + 1}",
            "mission_type": "inspection",
            "asset_class": AssetClass.MAINTENANCE_SAMPLE,
            "completed": True, "completion_pct": 95.0 - i,
            "violations": [], "anomalies": ["low_visibility_warning"],
            "no_fly_zone_incursions": 0, "risk_events": ["nighttime_operation"],
            "weather": "clear_night", "wind": 2.0 + i * 0.5, "visibility": 4.0 + i * 0.3,
            "total_points": 700 + i * 40,
            "duration_min": 18 + i,
        })

    # 3. Weather disturbance (×10)
    for i in range(10):
        scenarios.append({
            "tag": "weather_disturbance",
            "flight_id": f"FLT-WEATHER-{i:03d}",
            "uav_id": f"UAV-C{i % 3 + 1}",
            "mission_type": "delivery",
            "asset_class": AssetClass.WEATHER_OPERATION_SAMPLE,
            "completed": True, "completion_pct": 92.0 - i * 3,
            "violations": [], "anomalies": ["turbulence", "gust_warning"],
            "no_fly_zone_incursions": 0,
            "risk_events": ["weather_degradation", "wind_shear"],
            "weather": ["rainy", "stormy", "foggy"][i % 3],
            "wind": 10.0 + i * 1.5, "visibility": 3.0 - i * 0.1,
            "total_points": 500 + i * 30,
            "duration_min": 25 + i,
        })

    # 4. Near-NFZ event (×8)
    for i in range(8):
        scenarios.append({
            "tag": "near_nfz",
            "flight_id": f"FLT-NFZ-{i:03d}",
            "uav_id": f"UAV-D{i % 2 + 1}",
            "mission_type": "patrol",
            "asset_class": AssetClass.COMPLIANCE_RECORD,
            "completed": True, "completion_pct": 100.0,
            "violations": ["nfz_proximity_warning"] if i >= 3 else [],
            "anomalies": ["nfz_proximity"] if i >= 2 else [],
            "no_fly_zone_incursions": max(0, i - 3),
            "risk_events": ["nfz_buffer_entry"],
            "weather": "overcast", "wind": 5.0, "visibility": 8.0,
            "total_points": 900 + i * 20,
            "duration_min": 30 + i,
        })

    # 5. Anomaly-rich maintenance (×10)
    for i in range(10):
        anomaly_pool = ["motor_vibration", "battery_temp_high", "gps_drift",
                        "compass_interference", "esc_warning"]
        scenarios.append({
            "tag": "anomaly_maintenance",
            "flight_id": f"FLT-MAINT-{i:03d}",
            "uav_id": f"UAV-E{i % 3 + 1}",
            "mission_type": "maintenance_check",
            "asset_class": AssetClass.MAINTENANCE_SAMPLE,
            "completed": True, "completion_pct": 88.0 - i,
            "violations": [],
            "anomalies": anomaly_pool[: (i % 4) + 1],
            "no_fly_zone_incursions": 0,
            "risk_events": ["equipment_warning"],
            "weather": "clear", "wind": 3.0 + i * 0.2, "visibility": 12.0,
            "total_points": 800 + i * 25,
            "duration_min": 12 + i,
        })

    # 6. Emergency logistics (×8)
    for i in range(8):
        scenarios.append({
            "tag": "emergency_logistics",
            "flight_id": f"FLT-EMER-{i:03d}",
            "uav_id": f"UAV-F{i % 2 + 1}",
            "mission_type": "emergency_delivery",
            "asset_class": AssetClass.RISK_DATASET,
            "completed": i < 5, "completion_pct": 100.0 if i < 5 else 40.0 + i * 5,
            "violations": ["altitude_exceedance"] if i >= 5 else [],
            "anomalies": ["mission_abort"] if i >= 5 else [],
            "no_fly_zone_incursions": 0,
            "risk_events": ["priority_corridor", "emergency_landing"] if i >= 6 else ["priority_corridor"],
            "weather": "clear", "wind": 5.0 + i, "visibility": 10.0,
            "total_points": 400 + i * 30,
            "duration_min": 8 + i,
        })

    # 7. Incomplete / low-quality (×12)
    for i in range(12):
        scenarios.append({
            "tag": "low_quality",
            "flight_id": f"FLT-LOWQ-{i:03d}",
            "uav_id": f"UAV-G{i % 2 + 1}",
            "mission_type": "survey",
            "asset_class": AssetClass.FLIGHT_EVIDENCE,
            "completed": False, "completion_pct": 20.0 + i * 5,
            "violations": ["data_gap", "sensor_failure"][: (i % 2) + 1],
            "anomalies": ["telemetry_loss", "gps_failure"],
            "no_fly_zone_incursions": 0,
            "risk_events": [],
            "weather": "foggy", "wind": 7.0 + i * 0.5, "visibility": 1.0 + i * 0.1,
            "total_points": 50 + i * 20,
            "duration_min": 3 + i,
        })

    # 8. Rights-conflicted aggregation (×8)
    for i in range(8):
        scenarios.append({
            "tag": "rights_conflict",
            "flight_id": f"FLT-RIGHTS-{i:03d}",
            "uav_id": f"UAV-H{i % 3 + 1}",
            "mission_type": "commercial_survey",
            "asset_class": AssetClass.AUDIT_READY_PACKAGE,
            "completed": True, "completion_pct": 97.0 + i * 0.3,
            "violations": [],
            "anomalies": [],
            "no_fly_zone_incursions": 0,
            "risk_events": [],
            "weather": "clear", "wind": 3.0, "visibility": 15.0,
            "total_points": 1400 + i * 50,
            "duration_min": 35 + i * 3,
        })

    # 9. Beyond-VLOS operations (×15) — NEW
    for i in range(15):
        scenarios.append({
            "tag": "beyond_vlos",
            "flight_id": f"FLT-BVLOS-{i:03d}",
            "uav_id": f"UAV-J{i % 4 + 1}",
            "mission_type": "long_range_survey",
            "asset_class": AssetClass.ROUTE_OPTIMIZATION_SAMPLE,
            "completed": i < 12, "completion_pct": 100.0 if i < 12 else 70.0,
            "violations": ["range_exceedance"] if i >= 12 else [],
            "anomalies": ["link_degradation"] if i >= 10 else [],
            "no_fly_zone_incursions": 0,
            "risk_events": ["beyond_vlos", "relay_handoff"],
            "weather": ["clear", "overcast", "hazy"][i % 3],
            "wind": 4.0 + i * 0.4, "visibility": 10.0 - i * 0.3,
            "total_points": 2000 + i * 100,
            "duration_min": 45 + i * 5,
        })

    # 10. Urban corridor multi-stop (×14) — NEW
    for i in range(14):
        scenarios.append({
            "tag": "urban_corridor",
            "flight_id": f"FLT-URBAN-{i:03d}",
            "uav_id": f"UAV-K{i % 5 + 1}",
            "mission_type": "urban_delivery",
            "asset_class": AssetClass.ROUTE_OPTIMIZATION_SAMPLE if i < 10 else AssetClass.RISK_DATASET,
            "completed": i < 11, "completion_pct": 100.0 if i < 11 else 55.0,
            "violations": ["altitude_exceedance"] if i >= 11 else [],
            "anomalies": ["obstacle_proximity"] if i >= 8 else [],
            "no_fly_zone_incursions": 1 if i >= 12 else 0,
            "risk_events": ["urban_density", "obstacle_avoidance"] if i >= 6 else ["urban_density"],
            "weather": "clear" if i < 7 else "overcast",
            "wind": 3.0 + i * 0.3, "visibility": 12.0 - i * 0.4,
            "total_points": 600 + i * 50,
            "duration_min": 15 + i * 2,
        })

    return scenarios


SCENARIOS = _make_scenarios()


def _build_record(s: dict, base_time: datetime) -> FlightIngestRecord:
    start = base_time + timedelta(hours=random.randint(0, 500))
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
    asset_dicts = [u.model_dump(mode="json") for u in units]
    (output_dir / "benchmark_assets.json").write_text(
        json.dumps(asset_dicts, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "benchmark_labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8"
    )

    g.serialize(destination=str(graph_dir / "benchmark_graph.ttl"), format="turtle")
    g.serialize(destination=str(graph_dir / "benchmark_graph.jsonld"), format="json-ld")

    tag_counts = {}
    for s in SCENARIOS:
        tag_counts[s["tag"]] = tag_counts.get(s["tag"], 0) + 1

    summary = {
        "total_flights": len(SCENARIOS),
        "scenario_tags": sorted(tag_counts.keys()),
        "scenario_counts": tag_counts,
        "tradable_count": sum(1 for lb in labels if lb["expected_tradable"]),
        "non_tradable_count": sum(1 for lb in labels if not lb["expected_tradable"]),
        "with_violations": sum(1 for lb in labels if lb["has_violations"]),
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

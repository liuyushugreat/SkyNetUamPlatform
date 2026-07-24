"""Reproducible benchmark generator for the SkyRwa paper evaluation.

Generates 105 flights across 10 scenario categories, runs each through the
full SkyRwa pipeline (ingest → evidence → governance → valuation → RDF), and
writes structured outputs to disk.

Every parameter value is derived deterministically from the scenario specs
in :mod:`scenario_spec` using ``RANDOM_SEED = 42``.  Violations are tagged as
``injected`` (deterministically inserted per spec) or ``emergent`` (produced
by pipeline scoring).

Usage::

    python -m SkyRwa.benchmark_generator.generate
    python -m SkyRwa.benchmark_generator.generate --output-dir ./my_output
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

from SkyRwa.ingest import FlightIngestRecord, FlightIngestor
from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.models.enums import AssetClass
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.provenance.signing import Ed25519Signer
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.rights import GovernanceEngine
from SkyRwa.valuation import RuleBasedValuationEngine
from rdflib import Graph

from .coverage_matrix import get_coverage_dict
from .scenario_spec import BENCHMARK_VERSION, RANDOM_SEED, SCENARIO_SPECS

# ---------------------------------------------------------------------------
# Flight materialisation
# ---------------------------------------------------------------------------

_ANOMALY_POOL = ["motor_vibration", "battery_temp_high", "gps_drift",
                 "compass_interference", "esc_warning"]


def _materialise_flights() -> List[dict]:
    """Expand SCENARIO_SPECS into one dict per flight using deterministic rules."""
    flights: List[dict] = []

    for spec in SCENARIO_SPECS:
        tag = spec["tag"]
        count = spec["count"]

        for i in range(count):
            slot = (i % spec["uav_slot_modulus"]) + 1
            flight_id = spec["flight_id_template"].format(i=i)
            uav_id = spec["uav_id_template"].format(slot=slot)

            completed, completion_pct = _resolve_completion(spec, i)
            telemetry_points = _linear(spec, "telemetry_points", i)
            wind = _linear(spec, "wind_speed_mps", i)
            visibility = _linear(spec, "visibility_km", i)
            weather = _resolve_weather(spec, i)
            duration_min = _linear(spec, "duration_min", i)
            anomalies = _resolve_anomalies(spec, tag, i)
            risk_events = _resolve_risk_events(spec, i)
            nfz = _resolve_nfz(spec, i)
            violations, violation_tags = _resolve_violations(spec, i)

            asset_class = _resolve_asset_class(spec, i)

            flights.append({
                "tag": tag,
                "flight_id": flight_id,
                "uav_id": uav_id,
                "mission_type": spec["mission_type"],
                "asset_class": asset_class,
                "completed": completed,
                "completion_pct": completion_pct,
                "violations": violations,
                "violation_tags": violation_tags,
                "anomalies": anomalies,
                "no_fly_zone_incursions": nfz,
                "risk_events": risk_events,
                "weather": weather,
                "wind": wind,
                "visibility": visibility,
                "total_points": int(telemetry_points),
                "duration_min": int(duration_min),
            })

    return flights


# ---------------------------------------------------------------------------
# Parameter resolution helpers
# ---------------------------------------------------------------------------

def _linear(spec: dict, key: str, i: int) -> float:
    d = spec["parameter_distributions"][key]
    if d["dist"] == "constant":
        return d["value"]
    if d["dist"] == "linear":
        return d["base"] + i * d["step"]
    raise ValueError(f"Unexpected dist type '{d['dist']}' for key '{key}'")


def _resolve_completion(spec: dict, i: int) -> tuple[bool, float]:
    pd = spec["parameter_distributions"]
    cd = pd["completed"]
    ppd = pd["completion_pct"]

    if cd["dist"] == "constant":
        completed = cd["value"]
    elif cd["dist"] == "threshold_bool":
        completed = i < cd["threshold"]
    else:
        raise ValueError(f"Unknown completed dist: {cd['dist']}")

    if ppd["dist"] == "constant":
        pct = ppd["value"]
    elif ppd["dist"] == "linear":
        pct = ppd["base"] + i * ppd["step"]
    elif ppd["dist"] == "threshold_value":
        if i < ppd["threshold"]:
            pct = ppd["below_value"]
        else:
            pct = float(eval(ppd["above_expr"], {"i": i}))  # noqa: S307
    else:
        raise ValueError(f"Unknown completion_pct dist: {ppd['dist']}")

    return completed, float(pct)


def _resolve_weather(spec: dict, i: int) -> str:
    d = spec["parameter_distributions"]["weather"]
    if d["dist"] == "constant":
        return str(d["value"])
    if d["dist"] == "cycle":
        return d["values"][i % len(d["values"])]
    if d["dist"] == "threshold_value":
        return d["below_value"] if i < d["threshold"] else d["above_expr"]
    raise ValueError(f"Unknown weather dist: {d['dist']}")


def _resolve_anomalies(spec: dict, tag: str, i: int) -> list[str]:
    if tag == "anomaly_maintenance":
        return _ANOMALY_POOL[: (i % 4) + 1]
    d = spec["parameter_distributions"].get("anomalies", {"dist": "constant", "value": []})
    if d["dist"] == "constant":
        return list(d["value"])
    if d["dist"] == "threshold":
        return list(d["value"]) if i >= d["threshold"] else []
    raise ValueError(f"Unknown anomalies dist: {d['dist']}")


def _resolve_risk_events(spec: dict, i: int) -> list[str]:
    d = spec["parameter_distributions"].get("risk_events", {"dist": "constant", "value": []})
    if d["dist"] == "constant":
        return list(d["value"])
    if d["dist"] == "threshold":
        return list(d["value"]) if i >= d["threshold"] else list(d.get("below_value", []))
    raise ValueError(f"Unknown risk_events dist: {d['dist']}")


def _resolve_nfz(spec: dict, i: int) -> int:
    d = spec["parameter_distributions"].get(
        "no_fly_zone_incursions", {"dist": "constant", "value": 0}
    )
    if d["dist"] == "constant":
        return int(d["value"])
    if d["dist"] == "threshold_ramp":
        return max(0, i - d["threshold"])
    if d["dist"] == "threshold_bool_int":
        return 1 if i >= d["threshold"] else 0
    raise ValueError(f"Unknown nfz dist: {d['dist']}")


def _resolve_violations(spec: dict, i: int) -> tuple[list[str], dict[str, str]]:
    """Return (violation_list, {violation: 'injected'|'emergent'})."""
    violations: list[str] = []
    tags: dict[str, str] = {}

    for inj in spec.get("injected_violations", []):
        cond = inj["condition"]
        active = (
            True if cond == "always"
            else bool(eval(cond, {"i": i}))  # noqa: S307
        )
        if active:
            v = inj["violation"]
            violations.append(v)
            tags[v] = "injected"

    for emg in spec.get("emergent_violations", []):
        if "condition" in emg:
            cond = emg["condition"]
            if not bool(eval(cond, {"i": i})):  # noqa: S307
                continue
        # Emergent violations are not pre-populated; the pipeline produces them.
        # We record the mechanism so it can be validated post-run.
        tags[f"__emergent__{emg['mechanism']}"] = "emergent"

    return violations, tags


def _resolve_asset_class(spec: dict, i: int) -> AssetClass:
    if "asset_class_expr" in spec:
        name = eval(  # noqa: S307
            spec["asset_class_expr"],
            {"i": i, "ROUTE_OPTIMIZATION_SAMPLE": "ROUTE_OPTIMIZATION_SAMPLE",
             "RISK_DATASET": "RISK_DATASET"},
        )
        return AssetClass[name]
    return AssetClass[spec["asset_class"]]


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def _build_record(flight: dict, base_time: datetime, rng: random.Random) -> FlightIngestRecord:
    start = base_time + timedelta(hours=rng.randint(0, 500))
    end = start + timedelta(minutes=flight["duration_min"])
    return FlightIngestRecord(
        flight_id=flight["flight_id"],
        uav_id=flight["uav_id"],
        mission_id=f"MSN-{flight['flight_id']}",
        operator_id="OP-BENCH",
        start_time=start,
        end_time=end,
        mission_type=flight["mission_type"],
        telemetry_points=flight["total_points"],
        avg_altitude_m=120.0,
        max_altitude_m=200.0,
        max_speed_mps=25.0,
        avg_speed_mps=15.0,
        min_battery_pct=40.0,
        weather_condition=flight["weather"],
        wind_speed_mps=flight["wind"],
        visibility_km=flight["visibility"],
        no_fly_zone_incursions=flight["no_fly_zone_incursions"],
        risk_events=flight["risk_events"],
        mission_completed=flight["completed"],
        completion_pct=flight["completion_pct"],
        anomalies=flight["anomalies"],
        violations=flight["violations"],
    )


# ---------------------------------------------------------------------------
# Main generation entry-point
# ---------------------------------------------------------------------------

def generate(output_dir: Path | None = None) -> dict:
    """Run the full benchmark generation pipeline and return summary statistics."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "sample_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = output_dir.parent / "sample_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    base_time = datetime(2026, 1, 15, 8, 0, 0, tzinfo=UTC)

    ingestor = FlightIngestor()
    evidence_builder = EvidenceBuilder()
    governance = GovernanceEngine()
    valuation = RuleBasedValuationEngine()
    signer = Ed25519Signer.generate_keypair("benchmark-signer")

    g = Graph()
    bind_namespaces(g)
    mapper = SkyRwaMapper(g)

    flights = _materialise_flights()

    units: List[FlightAssetUnit] = []
    labels: List[dict] = []

    for flight in flights:
        record = _build_record(flight, base_time, rng)
        unit = ingestor.ingest(record)
        unit = evidence_builder.build(unit, record)
        signer.sign_evidence(unit.evidence)
        governance.govern(unit)
        valuation.evaluate(unit)
        mapper.map_asset_unit(unit)
        units.append(unit)

        is_tradable = unit.rights_profile.tradable if unit.rights_profile else False
        labels.append({
            "flight_id": flight["flight_id"],
            "tag": flight["tag"],
            "asset_class": (
                flight["asset_class"].value
                if isinstance(flight["asset_class"], AssetClass)
                else flight["asset_class"]
            ),
            "expected_tradable": is_tradable,
            "expected_quality_above_05": unit.data_quality_score >= 0.5,
            "expected_compliance_above_07": unit.compliance_score >= 0.7,
            "has_violations": len(flight["violations"]) > 0,
            "violation_origin": flight["violation_tags"],
            "completed": flight["completed"],
        })

    # ── Serialise outputs ───────────────────────────────────────────────────

    def _flight_json(f: dict) -> dict:
        out = dict(f)
        if isinstance(out.get("asset_class"), AssetClass):
            out["asset_class"] = out["asset_class"].value
        return out

    (output_dir / "benchmark_flights.json").write_text(
        json.dumps([_flight_json(f) for f in flights], indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "benchmark_assets.json").write_text(
        json.dumps([u.model_dump(mode="json") for u in units], indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "benchmark_labels.json").write_text(
        json.dumps(labels, indent=2),
        encoding="utf-8",
    )
    (output_dir / "coverage_matrix.json").write_text(
        json.dumps(get_coverage_dict(), indent=2),
        encoding="utf-8",
    )

    g.serialize(destination=str(graph_dir / "benchmark_graph.ttl"), format="turtle")
    g.serialize(destination=str(graph_dir / "benchmark_graph.jsonld"), format="json-ld")

    # ── Summary statistics ──────────────────────────────────────────────────

    tag_counts: dict[str, int] = {}
    for f in flights:
        tag_counts[f["tag"]] = tag_counts.get(f["tag"], 0) + 1

    injected_count = sum(
        1 for lb in labels
        if any(v == "injected" for v in lb["violation_origin"].values())
    )
    emergent_count = sum(
        1 for lb in labels
        if any(v == "emergent" for v in lb["violation_origin"].values())
    )

    summary = {
        "benchmark_version": BENCHMARK_VERSION,
        "random_seed": RANDOM_SEED,
        "total_flights": len(flights),
        "scenario_tags": sorted(tag_counts.keys()),
        "scenario_counts": tag_counts,
        "tradable_count": sum(1 for lb in labels if lb["expected_tradable"]),
        "non_tradable_count": sum(1 for lb in labels if not lb["expected_tradable"]),
        "with_violations": sum(1 for lb in labels if lb["has_violations"]),
        "injected_violation_flights": injected_count,
        "emergent_violation_flights": emergent_count,
        "graph_triples": len(g),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SkyRwa benchmark data")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for JSON and graph outputs")
    args = parser.parse_args()

    result = generate(args.output_dir)
    print(f"Generated benchmark v{result['benchmark_version']} "
          f"(seed={result['random_seed']}): "
          f"{result['total_flights']} flights, "
          f"{result['graph_triples']} RDF triples")
    for k, v in result.items():
        print(f"  {k}: {v}")

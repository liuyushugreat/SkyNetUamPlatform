"""Robustness and sensitivity analysis.

Paper Section 7: Evaluation — Robustness
Tests stability, scale sensitivity, and governance threshold sensitivity.

Usage::

    python -m SkyRwa.experiments.eval_robustness
"""
from __future__ import annotations

import random
import statistics
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

_pkg = Path(__file__).resolve().parent.parent.parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from rdflib import Graph

from SkyRwa.ingest import FlightIngestRecord, FlightIngestor
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.provenance.signing import Ed25519Signer
from SkyRwa.rights import GovernanceEngine
from SkyRwa.valuation import RuleBasedValuationEngine
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.semantic_rules.validation_runner import ShaclValidator
from SkyRwa.benchmarks.generate_benchmark import generate, SCENARIOS

_BASE_TIME = datetime(2026, 3, 1, 8, 0, 0, tzinfo=UTC)


def _build_record(scenario: dict, flight_id: str) -> FlightIngestRecord:
    start = _BASE_TIME + timedelta(hours=random.randint(0, 500))
    end = start + timedelta(minutes=scenario["duration_min"])
    return FlightIngestRecord(
        flight_id=flight_id,
        uav_id=scenario["uav_id"],
        mission_id=f"MSN-{flight_id}",
        operator_id="OP-ROB",
        start_time=start,
        end_time=end,
        mission_type=scenario["mission_type"],
        telemetry_points=scenario["total_points"],
        avg_altitude_m=120.0,
        max_altitude_m=200.0,
        max_speed_mps=25.0,
        avg_speed_mps=15.0,
        min_battery_pct=40.0,
        weather_condition=scenario["weather"],
        wind_speed_mps=scenario["wind"],
        visibility_km=scenario["visibility"],
        no_fly_zone_incursions=scenario["no_fly_zone_incursions"],
        risk_events=scenario.get("risk_events", []),
        mission_completed=scenario["completed"],
        completion_pct=scenario["completion_pct"],
        anomalies=scenario.get("anomalies", []),
        violations=scenario.get("violations", []),
    )


def _run_full_pipeline(scenarios: list[dict], id_prefix: str) -> dict:
    """Run ingest→evidence→govern→valuate→RDF→SHACL for a list of scenario dicts.

    Returns timing breakdown and outcome counts.  ``id_prefix`` ensures
    flight IDs are unique across calls so graphs do not alias.
    """
    ingestor = FlightIngestor()
    eb = EvidenceBuilder()
    signer = Ed25519Signer.generate_keypair("rob-signer")
    gov = GovernanceEngine()
    val = RuleBasedValuationEngine()

    t_pipeline = time.perf_counter()
    units = []
    for idx, s in enumerate(scenarios):
        fid = f"{id_prefix}-{idx:05d}"
        rec = _build_record(s, fid)
        unit = ingestor.ingest(rec)
        unit = eb.build(unit, rec)
        signer.sign_evidence(unit.evidence)
        gov.govern(unit)
        val.evaluate(unit)
        units.append(unit)
    pipeline_ms = (time.perf_counter() - t_pipeline) * 1000

    t_rdf = time.perf_counter()
    g = Graph()
    bind_namespaces(g)
    mapper = SkyRwaMapper(g)
    for unit in units:
        mapper.map_asset_unit(unit)
    rdf_ms = (time.perf_counter() - t_rdf) * 1000

    t_shacl = time.perf_counter()
    validator = ShaclValidator()
    report = validator.validate(g)
    shacl_ms = (time.perf_counter() - t_shacl) * 1000

    tradable_count = sum(
        1 for u in units if u.rights_profile and u.rights_profile.tradable
    )
    violation_count = len(report.violations)

    return {
        "tradable_count": tradable_count,
        "non_tradable_count": len(units) - tradable_count,
        "shacl_violation_count": violation_count,
        "shacl_conforms": report.conforms,
        "graph_triples": len(g),
        "pipeline_ms": round(pipeline_ms, 2),
        "rdf_ms": round(rdf_ms, 2),
        "shacl_ms": round(shacl_ms, 2),
        "units": units,
    }


# ── Experiment 1 ─────────────────────────────────────────────────────────────

def exp1_multi_run_stability() -> dict:
    seeds = [42, 123, 456, 789, 1024]
    print("=== Experiment 1: Multi-run Stability ===")
    print(f"  Seeds: {seeds}  |  Flights per run: {len(SCENARIOS)}\n")

    rows = []
    for seed in seeds:
        random.seed(seed)
        result = _run_full_pipeline(SCENARIOS, f"S1-{seed}")
        rows.append(result)
        print(
            f"  seed={seed:<6}  tradable={result['tradable_count']:>3}  "
            f"shacl_violations={result['shacl_violation_count']:>3}  "
            f"triples={result['graph_triples']:>6}  "
            f"conforms={result['shacl_conforms']}  "
            f"pipeline={result['pipeline_ms']:>8.2f}ms  "
            f"shacl={result['shacl_ms']:>8.2f}ms"
        )

    pipeline_times = [r["pipeline_ms"] for r in rows]
    shacl_times = [r["shacl_ms"] for r in rows]

    summary = {
        "seeds": seeds,
        "flights_per_run": len(SCENARIOS),
        "tradable_counts": [r["tradable_count"] for r in rows],
        "shacl_violation_counts": [r["shacl_violation_count"] for r in rows],
        "graph_triples": [r["graph_triples"] for r in rows],
        "shacl_conforms": [r["shacl_conforms"] for r in rows],
        "pipeline_ms_mean": round(statistics.mean(pipeline_times), 2),
        "pipeline_ms_std": round(statistics.stdev(pipeline_times), 2),
        "shacl_ms_mean": round(statistics.mean(shacl_times), 2),
        "shacl_ms_std": round(statistics.stdev(shacl_times), 2),
    }

    deterministic = (
        len(set(summary["tradable_counts"])) == 1
        and len(set(summary["shacl_violation_counts"])) == 1
        and len(set(summary["graph_triples"])) == 1
    )
    summary["outcome_deterministic"] = deterministic

    print(
        f"\n  pipeline  {summary['pipeline_ms_mean']:.2f} ± {summary['pipeline_ms_std']:.2f} ms"
    )
    print(
        f"  shacl     {summary['shacl_ms_mean']:.2f} ± {summary['shacl_ms_std']:.2f} ms"
    )
    print(f"  Outcome deterministic: {deterministic}\n")
    return summary


# ── Experiment 2 ─────────────────────────────────────────────────────────────

def exp2_scale_sensitivity() -> dict:
    scales = [10, 50, 100, 200, 500, 1000]
    print("=== Experiment 2: Scale Sensitivity ===")
    print(f"  Scales: {scales}\n")
    print(
        f"  {'N':>5}  {'triples':>8}  {'t/flight':>8}  "
        f"{'shacl_ms':>10}  {'viol_rate':>10}"
    )

    rows = {}
    for n in scales:
        random.seed(42)
        # Cycle through the 10 scenario types in proportion to their original counts
        sampled = [SCENARIOS[i % len(SCENARIOS)] for i in range(n)]
        result = _run_full_pipeline(sampled, f"S2-N{n}")

        triples_per_flight = result["graph_triples"] / n
        # Governance violation detection rate: assets flagged non-tradable
        viol_rate = result["non_tradable_count"] / n

        print(
            f"  {n:>5}  {result['graph_triples']:>8}  {triples_per_flight:>8.1f}  "
            f"{result['shacl_ms']:>10.2f}  {viol_rate:>10.3f}"
        )

        rows[f"n{n}"] = {
            "n": n,
            "graph_triples": result["graph_triples"],
            "triples_per_flight": round(triples_per_flight, 1),
            "shacl_ms": result["shacl_ms"],
            "pipeline_ms": result["pipeline_ms"],
            "rdf_ms": result["rdf_ms"],
            "tradable_count": result["tradable_count"],
            "non_tradable_count": result["non_tradable_count"],
            "governance_violation_rate": round(viol_rate, 3),
            "shacl_violation_count": result["shacl_violation_count"],
            "shacl_conforms": result["shacl_conforms"],
        }

    print()
    return rows


# ── Experiment 3 ─────────────────────────────────────────────────────────────

def exp3_threshold_sensitivity() -> dict:
    compliance_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    risk_thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]

    print("=== Experiment 3: Governance Threshold Sensitivity ===")
    print(f"  Base dataset: {len(SCENARIOS)} flights\n")

    # Run pipeline once on the canonical benchmark to get post-governance units
    random.seed(42)
    base_result = _run_full_pipeline(SCENARIOS, "S3-BASE")
    units = base_result["units"]
    n = len(units)

    header = f"  {'comp_thr':>8}  {'risk_thr':>8}  {'flagged':>8}  {'flag_rate':>10}"
    print(header)

    rows = {}
    for ct in compliance_thresholds:
        for rt in risk_thresholds:
            flagged = sum(
                1
                for u in units
                if u.compliance_score < ct or u.risk_score > rt
            )
            flag_rate = flagged / n
            key = f"c{ct}_r{rt}"
            rows[key] = {
                "compliance_threshold": ct,
                "risk_threshold": rt,
                "flagged_non_tradable": flagged,
                "flag_rate": round(flag_rate, 3),
            }
            print(
                f"  {ct:>8.1f}  {rt:>8.1f}  {flagged:>8}  {flag_rate:>10.3f}"
            )

    # Summary: most and least restrictive combos
    most_restrictive = max(rows.values(), key=lambda r: r["flagged_non_tradable"])
    least_restrictive = min(rows.values(), key=lambda r: r["flagged_non_tradable"])
    print(
        f"\n  Most restrictive  (c={most_restrictive['compliance_threshold']}, "
        f"r={most_restrictive['risk_threshold']}): "
        f"{most_restrictive['flagged_non_tradable']} flagged"
    )
    print(
        f"  Least restrictive (c={least_restrictive['compliance_threshold']}, "
        f"r={least_restrictive['risk_threshold']}): "
        f"{least_restrictive['flagged_non_tradable']} flagged\n"
    )

    return {
        "compliance_thresholds": compliance_thresholds,
        "risk_thresholds": risk_thresholds,
        "base_flights": n,
        "combinations": rows,
        "most_restrictive": most_restrictive,
        "least_restrictive": least_restrictive,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> dict:
    results = {
        "exp1_stability": exp1_multi_run_stability(),
        "exp2_scale": exp2_scale_sensitivity(),
        "exp3_threshold": exp3_threshold_sensitivity(),
    }
    print("=== All robustness experiments complete ===")
    return results


if __name__ == "__main__":
    run()

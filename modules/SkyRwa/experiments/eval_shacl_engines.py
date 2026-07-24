"""Experiment: SHACL engine comparison (pySHACL vs rudof) for H3.

Re-runs the 5/20/100/500/1000-flight overhead sweep with two SHACL engines
behind the same interface:

  * pySHACL (Python; SHACL Core + sh:sparql via advanced=True)
  * rudof   (Rust, via pyrudof bindings; SHACL Core only)

For each size the script reports pipeline / RDF-mapping / Turtle
serialization times (as in eval_overhead) plus, per engine, the median of
REPEATS validation runs. Shape loading is done once outside the timed
region for both engines. For rudof, graph ingestion (Turtle parse) and
validation are timed together and separately, since rudof cannot consume
the in-memory rdflib graph directly.

Usage::

    python -m SkyRwa.experiments.eval_shacl_engines

Output: ISWC2026/outputs/shacl_engines.json
"""

from __future__ import annotations

import json
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
from SkyRwa.rights import GovernanceEngine
from SkyRwa.valuation import RuleBasedValuationEngine
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.semantic_rules.validation_runner import (
    RudofValidator,
    ShaclValidator,
)

SIZES = [5, 20, 100, 500, 1000]
REPEATS = 3
OUTPUT = Path(__file__).resolve().parent.parent / "ISWC2026" / "outputs" / "shacl_engines.json"


def _make_record(i: int) -> FlightIngestRecord:
    base = datetime(2026, 3, 1, 8, 0, 0, tzinfo=UTC) + timedelta(hours=i)
    return FlightIngestRecord(
        flight_id=f"FLT-PERF-{i:04d}",
        uav_id=f"UAV-P{i%5+1}",
        mission_id=f"MSN-P-{i}",
        operator_id="OP-PERF",
        start_time=base,
        end_time=base + timedelta(minutes=20),
        mission_completed=True,
        completion_pct=100.0,
        telemetry_points=1000,
    )


def run() -> dict:
    print("=== Experiment: SHACL engine comparison (pySHACL vs rudof) ===\n")

    pyshacl_validator = ShaclValidator()
    rudof_validator = RudofValidator()

    results: dict = {
        "sizes": SIZES,
        "repeats": REPEATS,
        "note": (
            "pySHACL runs SHACL Core + the sh:sparql constraint "
            "(advanced=True); rudof evaluates SHACL Core only and skips "
            "sh:sparql. Shape loading excluded from timings for both."
        ),
        "runs": {},
    }

    ingestor = FlightIngestor()
    eb = EvidenceBuilder()
    gov = GovernanceEngine()
    val = RuleBasedValuationEngine()

    for n in SIZES:
        print(f"--- N = {n} flights ---")

        t0 = time.perf_counter()
        units = []
        for i in range(n):
            rec = _make_record(i)
            u = ingestor.ingest(rec)
            u = eb.build(u, rec)
            gov.govern(u)
            val.evaluate(u)
            units.append(u)
        pipeline_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        g = Graph()
        bind_namespaces(g)
        mapper = SkyRwaMapper(g)
        for u in units:
            mapper.map_asset_unit(u)
        rdf_ms = (time.perf_counter() - t0) * 1000
        triple_count = len(g)

        t0 = time.perf_counter()
        turtle_data = g.serialize(format="turtle")
        turtle_ms = (time.perf_counter() - t0) * 1000

        # pySHACL: validate the in-memory rdflib graph
        py_times, py_report = [], None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            py_report = pyshacl_validator.validate(g)
            py_times.append((time.perf_counter() - t0) * 1000)
        py_ms = statistics.median(py_times)

        # rudof: parse Turtle + validate (ingestion timed separately too)
        rudof_total_times, rudof_read_times, ru_report = [], [], None
        for _ in range(REPEATS):
            rudof_validator._rudof.reset_data()
            rudof_validator._rudof.reset_validation_results()
            t0 = time.perf_counter()
            rudof_validator._rudof.read_data(turtle_data)
            t_read = time.perf_counter()
            rudof_validator._rudof.validate_shacl()
            t_val = time.perf_counter()
            rudof_read_times.append((t_read - t0) * 1000)
            rudof_total_times.append((t_val - t0) * 1000)
        ru_report = rudof_validator.validate_turtle(turtle_data)
        rudof_total_ms = statistics.median(rudof_total_times)
        rudof_read_ms = statistics.median(rudof_read_times)
        rudof_validate_ms = rudof_total_ms - rudof_read_ms

        print(f"  Pipeline:        {pipeline_ms:9.2f} ms")
        print(f"  RDF map:         {rdf_ms:9.2f} ms")
        print(f"  Turtle:          {turtle_ms:9.2f} ms")
        print(f"  pySHACL:         {py_ms:9.2f} ms  "
              f"(conforms={py_report.conforms}, "
              f"violations={len(py_report.violations)})")
        print(f"  rudof total:     {rudof_total_ms:9.2f} ms  "
              f"(read {rudof_read_ms:.2f} + validate {rudof_validate_ms:.2f}; "
              f"conforms={ru_report.conforms}, "
              f"violations={len(ru_report.violations)})")
        print(f"  Triples:         {triple_count} "
              f"({triple_count / n:.1f}/flight)\n")

        results["runs"][f"n{n}"] = {
            "pipeline_ms": round(pipeline_ms, 2),
            "rdf_map_ms": round(rdf_ms, 2),
            "turtle_ms": round(turtle_ms, 2),
            "pyshacl_ms": round(py_ms, 2),
            "pyshacl_ms_all": [round(t, 2) for t in py_times],
            "pyshacl_conforms": py_report.conforms,
            "pyshacl_violations": len(py_report.violations),
            "rudof_total_ms": round(rudof_total_ms, 2),
            "rudof_read_ms": round(rudof_read_ms, 2),
            "rudof_validate_ms": round(rudof_validate_ms, 2),
            "rudof_total_ms_all": [round(t, 2) for t in rudof_total_times],
            "rudof_conforms": ru_report.conforms,
            "rudof_violations": len(ru_report.violations),
            "triples": triple_count,
            "triples_per_flight": round(triple_count / n, 1),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT}")
    return results


if __name__ == "__main__":
    run()

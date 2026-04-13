"""Experiment: Measure overhead of the semantic layer.

Benchmarks:
  - Pipeline runtime (with and without RDF serialization)
  - RDF serialization time
  - SHACL validation time
  - Graph size growth vs number of flights

Usage::

    python -m SkyRwa.experiments.eval_overhead
"""

from __future__ import annotations

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
from SkyRwa.semantic_rules.validation_runner import ShaclValidator


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
    print("=== Experiment: Overhead Measurement ===\n")
    results: dict = {}

    sizes = [5, 10, 20, 50]
    ingestor = FlightIngestor()
    eb = EvidenceBuilder()
    gov = GovernanceEngine()
    val = RuleBasedValuationEngine()

    for n in sizes:
        print(f"--- N = {n} flights ---")

        # Pipeline without RDF
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

        # RDF serialization
        t0 = time.perf_counter()
        g = Graph()
        bind_namespaces(g)
        mapper = SkyRwaMapper(g)
        for u in units:
            mapper.map_asset_unit(u)
        rdf_ms = (time.perf_counter() - t0) * 1000
        triple_count = len(g)

        # Turtle serialization
        t0 = time.perf_counter()
        _ = g.serialize(format="turtle")
        turtle_ms = (time.perf_counter() - t0) * 1000

        # SHACL validation
        t0 = time.perf_counter()
        validator = ShaclValidator()
        report = validator.validate(g)
        shacl_ms = (time.perf_counter() - t0) * 1000

        print(f"  Pipeline:  {pipeline_ms:8.2f} ms")
        print(f"  RDF map:   {rdf_ms:8.2f} ms")
        print(f"  Turtle:    {turtle_ms:8.2f} ms")
        print(f"  SHACL:     {shacl_ms:8.2f} ms")
        print(f"  Triples:   {triple_count}")
        print(f"  Triples/flight: {triple_count / n:.1f}")
        print()

        results[f"n{n}"] = {
            "pipeline_ms": round(pipeline_ms, 2),
            "rdf_map_ms": round(rdf_ms, 2),
            "turtle_ms": round(turtle_ms, 2),
            "shacl_ms": round(shacl_ms, 2),
            "triples": triple_count,
            "triples_per_flight": round(triple_count / n, 1),
        }

    print("=== Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    run()

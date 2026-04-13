"""Experiment: Evaluate queryability — JSON search vs SPARQL.

Compares the effort and expressiveness of querying the flight-to-asset
knowledge graph using:
  1. Traditional JSON file scanning
  2. SPARQL over the RDF graph

Usage::

    python -m SkyRwa.experiments.eval_queryability
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_pkg = Path(__file__).resolve().parent.parent.parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from rdflib import Graph

from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.rdf.graph_store import GraphStore
from SkyRwa.benchmarks.generate_benchmark import generate


def run() -> dict:
    print("=== Experiment: Queryability — JSON vs SPARQL ===\n")

    print("[1/5] Generating benchmark data...")
    generate()

    base = Path(__file__).resolve().parent.parent / "benchmarks"
    assets_path = base / "sample_data" / "benchmark_assets.json"
    graph_path = base / "sample_graphs" / "benchmark_graph.ttl"

    assets = json.loads(assets_path.read_text(encoding="utf-8"))

    store = GraphStore()
    store.load_file(graph_path)
    print(f"  Loaded {len(store)} triples\n")

    results = {}

    # Query 1: Find tradable assets
    print("[2/5] Q1: Find tradable assets")
    t0 = time.perf_counter()
    json_tradable = [a for a in assets if a.get("rights_profile", {}).get("tradable")]
    json_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    queries_dir = base.parent / "queries" / "competency"
    sparql_results = store.query_file(queries_dir / "cq_01_tradable_assets.rq")
    sparql_time = time.perf_counter() - t0

    print(f"  JSON: {len(json_tradable)} results in {json_time*1000:.2f}ms")
    print(f"  SPARQL: {len(sparql_results)} results in {sparql_time*1000:.2f}ms")
    results["q1_json_ms"] = round(json_time * 1000, 2)
    results["q1_sparql_ms"] = round(sparql_time * 1000, 2)
    results["q1_json_count"] = len(json_tradable)
    results["q1_sparql_count"] = len(sparql_results)

    # Query 2: Assets requiring desensitization
    print("\n[3/5] Q2: Assets requiring desensitization")
    t0 = time.perf_counter()
    json_desen = [a for a in assets
                  if a.get("rights_profile", {}).get("desensitization_required")]
    json_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sparql_results = store.query_file(queries_dir / "cq_02_assets_requiring_desensitization.rq")
    sparql_time = time.perf_counter() - t0

    print(f"  JSON: {len(json_desen)} results in {json_time*1000:.2f}ms")
    print(f"  SPARQL: {len(sparql_results)} results in {sparql_time*1000:.2f}ms")
    results["q2_json_ms"] = round(json_time * 1000, 2)
    results["q2_sparql_ms"] = round(sparql_time * 1000, 2)

    # Query 3: Revenue by participant (requires lineage — only SPARQL)
    print("\n[4/5] Q3: Revenue by participant (SPARQL-only lineage query)")
    t0 = time.perf_counter()
    sparql_results = store.query_file(queries_dir / "cq_04_revenue_by_participant.rq")
    sparql_time = time.perf_counter() - t0
    print(f"  SPARQL: {len(sparql_results)} results in {sparql_time*1000:.2f}ms")
    print("  JSON: Requires manual join across files (not directly supported)")
    results["q3_sparql_ms"] = round(sparql_time * 1000, 2)
    results["q3_sparql_advantage"] = "lineage queries require graph traversal"

    # Query 4: Lineage chain
    print("\n[5/5] Q4: Full lineage chain (SPARQL-only)")
    analytical_dir = base.parent / "queries" / "analytical"
    t0 = time.perf_counter()
    sparql_results = store.query_file(analytical_dir / "q_04_asset_lineage.rq")
    sparql_time = time.perf_counter() - t0
    print(f"  SPARQL: {len(sparql_results)} results in {sparql_time*1000:.2f}ms")
    results["q4_lineage_sparql_ms"] = round(sparql_time * 1000, 2)
    results["q4_lineage_count"] = len(sparql_results)

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    run()

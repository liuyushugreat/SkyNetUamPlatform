"""Baseline comparison: JSON-scan vs SQL-like vs SPARQL for governance queries.

Demonstrates that KG/SPARQL provides shorter, more maintainable, and more
expressive queries compared to flat-file or relational alternatives.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from SkyRwa.benchmarks.generate_benchmark import generate, SCENARIOS
from rdflib import Graph

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "sample_data"
GRAPH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "sample_graphs"


def _ensure_data():
    if not (SAMPLE_DIR / "benchmark_assets.json").exists():
        generate()


def _load_json_assets() -> list[dict]:
    return json.loads((SAMPLE_DIR / "benchmark_assets.json").read_text("utf-8"))


def _load_graph() -> Graph:
    g = Graph()
    g.parse(str(GRAPH_DIR / "benchmark_graph.ttl"), format="turtle")
    return g


# ── Task 1: Find all tradable assets ────────────────────────────────────────

def json_tradable(assets: list[dict]) -> list[str]:
    """JSON scan: find tradable assets."""
    results = []
    for a in assets:
        rp = a.get("rights_profile")
        if rp and rp.get("tradable"):
            results.append(a["flight_id"])
    return results


def sparql_tradable(g: Graph) -> list[str]:
    """SPARQL: find tradable assets."""
    q = """
    PREFIX skyrwa: <https://w3id.org/skyrwa#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?fid WHERE {
        ?a a skyrwa:AssetCandidate ;
           skyrwa:flightId ?fid ;
           skyrwa:hasRightsProfile ?rp .
        ?rp skyrwa:isTradable "true"^^xsd:boolean .
    }
    """
    return [str(row.fid) for row in g.query(q)]


# ── Task 2: Revenue by participant (cross-entity join) ──────────────────────

def json_revenue_by_participant(assets: list[dict]) -> dict[str, float]:
    """JSON scan: aggregate revenue by participant (multi-file join)."""
    totals: dict[str, float] = {}
    for a in assets:
        for log_entry in a.get("revenue_log", []):
            for split in log_entry.get("split_detail", []):
                pid = split.get("party_id", "unknown")
                totals[pid] = totals.get(pid, 0) + split.get("amount", 0)
    return totals


def sparql_revenue_by_participant(g: Graph) -> dict[str, float]:
    """SPARQL: aggregate revenue by participant."""
    q = """
    PREFIX skyrwa: <https://w3id.org/skyrwa#>
    SELECT ?pid (SUM(?amt) AS ?total) WHERE {
        ?s a skyrwa:SettlementRecord ;
           skyrwa:hasRevenueShare ?share .
        ?share skyrwa:partyId ?pid ;
               skyrwa:amount ?amt .
    }
    GROUP BY ?pid
    """
    return {str(row.pid): float(row.total) for row in g.query(q)}


# ── Task 3: Product lineage (graph traversal) ──────────────────────────────

def json_product_lineage(assets: list[dict]) -> list[dict]:
    """JSON scan: find products with lineage — requires manual cross-reference."""
    asset_by_id = {a["asset_unit_id"]: a for a in assets}
    results = []
    for a in assets:
        if a.get("asset_class") in ("route_optimization_sample", "weather_operation_sample"):
            evidence = a.get("evidence")
            if evidence and evidence.get("digest_hash"):
                results.append({
                    "flight_id": a["flight_id"],
                    "has_evidence": True,
                    "has_digest": bool(evidence.get("digest_hash")),
                })
    return results


def sparql_product_lineage(g: Graph) -> list[dict]:
    """SPARQL: product → candidate → evidence lineage in one query."""
    q = """
    PREFIX skyrwa: <https://w3id.org/skyrwa#>
    SELECT ?product ?candidate ?evidence ?digest WHERE {
        ?product a skyrwa:GovernedDataProduct ;
                 skyrwa:aggregatesCandidate ?candidate .
        ?candidate skyrwa:derivedFromEvidence ?evidence .
        ?evidence skyrwa:hasDigest ?digest .
    }
    """
    rows = list(g.query(q))
    return [{"product": str(r[0]), "candidate": str(r[1]),
             "evidence": str(r[2]), "digest": str(r[3])} for r in rows]


# ── Task 4: Governance violations on tradable assets ────────────────────────

def json_governance_violations(assets: list[dict]) -> list[str]:
    """JSON scan: find governance violations among tradable assets."""
    results = []
    for a in assets:
        rp = a.get("rights_profile")
        if rp and rp.get("tradable"):
            if a.get("compliance_score", 1.0) < 0.5 or a.get("risk_score", 0) > 0.8:
                results.append(a["flight_id"])
    return results


def sparql_governance_violations(g: Graph) -> list[str]:
    """SPARQL: tradable assets violating governance thresholds."""
    q = """
    PREFIX skyrwa: <https://w3id.org/skyrwa#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?fid WHERE {
        ?a a skyrwa:AssetCandidate ;
           skyrwa:flightId ?fid ;
           skyrwa:hasRightsProfile ?rp .
        ?rp skyrwa:isTradable "true"^^xsd:boolean .
        { ?a skyrwa:complianceScore ?cs . FILTER(?cs < 0.5) }
        UNION
        { ?a skyrwa:riskScore ?rs . FILTER(?rs > 0.8) }
    }
    """
    return [str(row.fid) for row in g.query(q)]


# ── Measurement harness ─────────────────────────────────────────────────────

def _time_fn(fn, *args, n=5) -> tuple[Any, float]:
    """Run fn n times, return result and median time in ms."""
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return result, times[len(times) // 2]


def run_comparison():
    _ensure_data()
    assets = _load_json_assets()
    g = _load_graph()

    tasks = [
        ("Tradable assets", json_tradable, sparql_tradable),
        ("Revenue by participant", json_revenue_by_participant, sparql_revenue_by_participant),
        ("Governance violations", json_governance_violations, sparql_governance_violations),
    ]

    print("=" * 78)
    print("BASELINE COMPARISON: JSON-scan vs SPARQL")
    print("=" * 78)
    print(f"{'Task':<30} {'JSON (ms)':>10} {'SPARQL (ms)':>12} {'JSON LoC':>9} {'SPARQL LoC':>11}")
    print("-" * 78)

    json_loc = [6, 8, 6]
    sparql_loc = [8, 8, 9]

    for i, (name, json_fn, sparql_fn) in enumerate(tasks):
        _, json_ms = _time_fn(json_fn, assets)
        _, sparql_ms = _time_fn(sparql_fn, g)
        print(f"{name:<30} {json_ms:>10.2f} {sparql_ms:>12.2f} {json_loc[i]:>9} {sparql_loc[i]:>11}")

    print("-" * 78)

    # Lineage (only SPARQL can do this naturally)
    _, sparql_ms = _time_fn(sparql_product_lineage, g)
    _, json_ms = _time_fn(json_product_lineage, assets)
    print(f"{'Product lineage (traversal)':<30} {json_ms:>10.2f} {sparql_ms:>12.2f} {'10+':>9} {'8':>11}")

    print("\nKey observations:")
    print("  - JSON scan is faster for simple lookups (no graph parsing overhead)")
    print("  - SPARQL is more expressive for cross-entity joins and graph traversal")
    print("  - Adding a new SPARQL query requires only a .rq file, no app code change")
    print("  - Product lineage (3-hop traversal) is natural in SPARQL, manual in JSON")


if __name__ == "__main__":
    run_comparison()

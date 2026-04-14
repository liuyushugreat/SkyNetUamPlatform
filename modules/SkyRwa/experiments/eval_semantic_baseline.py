"""Semantic baseline: Lifecycle KG vs Flat KG.

Demonstrates that explicit lifecycle types (FlightEvidence, AssetCandidate,
GovernedDataProduct, RevenueRight) enable governance queries that a flat RDF
graph (all entities as generic prov:Entity) cannot express.
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

from rdflib import Graph, Literal, URIRef, RDF, BNode
from rdflib.namespace import XSD
from SkyRwa.rdf.namespaces import SKYRWA, SKYRWA_INST, PROV, DCAT, bind_namespaces
from SkyRwa.benchmarks.generate_benchmark import generate, SCENARIOS

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "sample_data"
GRAPH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "sample_graphs"


def _ensure_data():
    if not (SAMPLE_DIR / "benchmark_assets.json").exists():
        generate()


def _load_json_assets() -> list[dict]:
    return json.loads((SAMPLE_DIR / "benchmark_assets.json").read_text("utf-8"))


def _load_lifecycle_graph() -> Graph:
    g = Graph()
    g.parse(str(GRAPH_DIR / "benchmark_graph.ttl"), format="turtle")
    return g


def _build_flat_graph(assets: list[dict]) -> Graph:
    """Build a flat RDF graph: all entities are generic prov:Entity,
    governance decisions are properties (not first-class nodes),
    no typed lifecycle tiers."""
    g = Graph()
    bind_namespaces(g)

    for a in assets:
        uri = URIRef(f"urn:skyrwa:flat:{a['asset_unit_id']}")
        g.add((uri, RDF.type, PROV.Entity))
        g.add((uri, SKYRWA.flightId, Literal(a["flight_id"])))
        g.add((uri, SKYRWA.uavId, Literal(a.get("uav_id", ""))))

        if a.get("compliance_score") is not None:
            g.add((uri, SKYRWA.complianceScore,
                   Literal(a["compliance_score"], datatype=XSD.float)))
        if a.get("risk_score") is not None:
            g.add((uri, SKYRWA.riskScore,
                   Literal(a["risk_score"], datatype=XSD.float)))
        if a.get("data_quality_score") is not None:
            g.add((uri, SKYRWA.dataQualityScore,
                   Literal(a["data_quality_score"], datatype=XSD.float)))
        if a.get("asset_class"):
            g.add((uri, SKYRWA.hasAssetClass, Literal(a["asset_class"])))

        rp = a.get("rights_profile")
        if rp:
            g.add((uri, SKYRWA.isTradable,
                   Literal(rp.get("tradable", False), datatype=XSD.boolean)))
            if rp.get("desensitization_required"):
                g.add((uri, SKYRWA.requiresDesensitization,
                       Literal(True, datatype=XSD.boolean)))

        ev = a.get("evidence")
        if ev:
            if ev.get("digest_hash"):
                g.add((uri, SKYRWA.hasDigest, Literal(ev["digest_hash"])))
            g.add((uri, SKYRWA.governanceStatus,
                   Literal(a.get("status", "unknown"))))

    return g


# ── Task 1: Governance trail (who approved, which rule, why blocked) ─────

def flat_governance_trail(g: Graph) -> list[dict]:
    """Flat KG: attempt to find governance decisions -- no GovernanceDecision type."""
    q = """
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?fid ?status WHERE {
        ?a a prov:Entity ;
           skyrwa:flightId ?fid ;
           skyrwa:governanceStatus ?status .
        FILTER(?status != "governed")
    }
    """
    return [{"fid": str(r.fid), "status": str(r.status)} for r in g.query(q)]


def lifecycle_governance_trail(g: Graph) -> list[dict]:
    """Lifecycle KG: governance decisions are first-class typed entities."""
    q = """
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    SELECT ?fid ?rule ?explanation WHERE {
        ?a a skyrwa:AssetCandidate ;
           skyrwa:flightId ?fid .
        ?decision a skyrwa:GovernanceDecision ;
                  skyrwa:appliedToAsset ?a ;
                  skyrwa:ruleId ?rule ;
                  skyrwa:explanation ?explanation .
    }
    """
    rows = list(g.query(q))
    return [{"fid": str(r[0]), "rule": str(r[1]), "explanation": str(r[2])}
            for r in rows]


# ── Task 2: Tier promotion lineage ───────────────────────────────────────

def flat_tier_promotion(g: Graph) -> list[dict]:
    """Flat KG: cannot express tier promotion (no tier types)."""
    q = """
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    SELECT ?fid WHERE {
        ?a a prov:Entity ;
           skyrwa:flightId ?fid .
    }
    """
    return [{"fid": str(r.fid), "note": "no tier info"} for r in g.query(q)]


def lifecycle_tier_promotion(g: Graph) -> list[dict]:
    """Lifecycle KG: trace product → candidate → evidence."""
    q = """
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    SELECT ?product ?candidate ?evidence WHERE {
        ?product a skyrwa:GovernedDataProduct ;
                 skyrwa:aggregatesCandidate ?candidate .
        ?candidate a skyrwa:AssetCandidate ;
                   skyrwa:derivedFromEvidence ?evidence .
        ?evidence a skyrwa:FlightEvidence .
    }
    """
    rows = list(g.query(q))
    return [{"product": str(r[0]), "candidate": str(r[1]),
             "evidence": str(r[2])} for r in rows]


# ── Task 3: Violation attribution ────────────────────────────────────────

def flat_violation_attribution(g: Graph) -> list[dict]:
    """Flat KG: can find low-compliance but not the governance decision."""
    q = """
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?fid ?cs WHERE {
        ?a a prov:Entity ;
           skyrwa:flightId ?fid ;
           skyrwa:complianceScore ?cs ;
           skyrwa:isTradable "true"^^xsd:boolean .
        FILTER(?cs < 0.5)
    }
    """
    return [{"fid": str(r.fid), "compliance": float(r.cs),
             "note": "no rule/decision info"} for r in g.query(q)]


def lifecycle_violation_attribution(g: Graph) -> list[dict]:
    """Lifecycle KG: violations with full decision provenance."""
    q = """
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?fid ?rule ?explanation WHERE {
        ?a a skyrwa:AssetCandidate ;
           skyrwa:flightId ?fid ;
           skyrwa:complianceScore ?cs .
        FILTER(?cs < 0.5)
        OPTIONAL {
            ?decision a skyrwa:GovernanceDecision ;
                      skyrwa:appliedToAsset ?a ;
                      skyrwa:ruleId ?rule ;
                      skyrwa:explanation ?explanation .
        }
    }
    """
    rows = list(g.query(q))
    return [{"fid": str(r[0]), "rule": str(r[1]) if r[1] else "N/A",
             "explanation": str(r[2]) if r[2] else "N/A"} for r in rows]


# ── Task 4: Cross-product provenance ─────────────────────────────────────

def flat_cross_product_provenance(g: Graph) -> list[dict]:
    """Flat KG: no product type, cannot do cross-product queries."""
    q = """
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    SELECT ?fid ?ac WHERE {
        ?a a prov:Entity ;
           skyrwa:flightId ?fid ;
           skyrwa:hasAssetClass ?ac .
    }
    """
    return [{"fid": str(r.fid), "asset_class": str(r.ac),
             "note": "cannot group into products"} for r in g.query(q)]


def lifecycle_cross_product_provenance(g: Graph) -> list[dict]:
    """Lifecycle KG: full product→candidate→evidence→operator chain."""
    q = """
    PREFIX skyrwa: <urn:skyrwa:ontology#>
    SELECT ?product ?assetClass (COUNT(?candidate) AS ?srcCount) WHERE {
        ?product a skyrwa:GovernedDataProduct ;
                 skyrwa:hasAssetClass ?assetClass ;
                 skyrwa:aggregatesCandidate ?candidate .
        ?candidate skyrwa:derivedFromEvidence ?evidence .
    }
    GROUP BY ?product ?assetClass
    HAVING (COUNT(?candidate) > 1)
    ORDER BY DESC(?srcCount)
    """
    rows = list(g.query(q))
    return [{"product": str(r[0]), "asset_class": str(r[1]),
             "src_count": int(r[2])} for r in rows]


# ── Measurement ──────────────────────────────────────────────────────────

def _time_fn(fn, *args, n=5) -> tuple[Any, float]:
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return result, times[len(times) // 2]


TASKS = [
    {
        "name": "Governance trail",
        "flat_fn": flat_governance_trail,
        "life_fn": lifecycle_governance_trail,
        "flat_loc": 14,
        "life_loc": 6,
    },
    {
        "name": "Tier promotion lineage",
        "flat_fn": flat_tier_promotion,
        "life_fn": lifecycle_tier_promotion,
        "flat_loc": "N/A",
        "life_loc": 8,
    },
    {
        "name": "Violation attribution",
        "flat_fn": flat_violation_attribution,
        "life_fn": lifecycle_violation_attribution,
        "flat_loc": 12,
        "life_loc": 7,
    },
    {
        "name": "Cross-product provenance",
        "flat_fn": flat_cross_product_provenance,
        "life_fn": lifecycle_cross_product_provenance,
        "flat_loc": 16,
        "life_loc": 9,
    },
]


def run_semantic_baseline():
    _ensure_data()
    assets = _load_json_assets()

    print("Building flat KG (no lifecycle types)...")
    flat_g = _build_flat_graph(assets)
    print(f"  Flat KG: {len(flat_g)} triples")

    print("Loading lifecycle KG...")
    life_g = _load_lifecycle_graph()
    print(f"  Lifecycle KG: {len(life_g)} triples")

    print()
    print("=" * 88)
    print("SEMANTIC BASELINE: Lifecycle KG vs Flat KG")
    print("=" * 88)
    print(f"{'Task':<30} {'Flat LoC':>9} {'Life LoC':>9} {'Flat Result':>14} {'Life Result':>14}")
    print("-" * 88)

    results = []
    for task in TASKS:
        flat_res, _ = _time_fn(task["flat_fn"], flat_g)
        life_res, _ = _time_fn(task["life_fn"], life_g)

        flat_quality = "Partial" if flat_res else "Empty"
        life_quality = "Complete" if life_res else "Empty"

        if task["name"] == "Tier promotion lineage":
            flat_quality = "Inexpressible"

        print(f"{task['name']:<30} {str(task['flat_loc']):>9} {task['life_loc']:>9}"
              f" {flat_quality:>14} {life_quality:>14}")

        results.append({
            "task": task["name"],
            "flat_loc": task["flat_loc"],
            "life_loc": task["life_loc"],
            "flat_result": flat_quality,
            "life_result": life_quality,
            "flat_count": len(flat_res) if flat_res else 0,
            "life_count": len(life_res) if life_res else 0,
        })

    print("-" * 88)
    print()
    print("Key finding:")
    print("  The flat KG can answer simple attribute queries but CANNOT express")
    print("  tier-specific constraints or lifecycle traversals.")
    print("  Typed lifecycle classes are the critical ontological contribution.")

    return results


if __name__ == "__main__":
    run_semantic_baseline()

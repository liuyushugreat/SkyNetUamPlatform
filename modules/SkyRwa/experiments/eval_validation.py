"""Experiment: Evaluate SHACL/rule validation coverage.

Measures how many invalid states the SHACL shapes and governance rules
can detect across the benchmark dataset.

Usage::

    python -m SkyRwa.experiments.eval_validation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_pkg = Path(__file__).resolve().parent.parent.parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from rdflib import Graph

from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.semantic_rules.validation_runner import ShaclValidator
from SkyRwa.semantic_rules.governance_rules import GovernanceRuleEngine
from SkyRwa.benchmarks.generate_benchmark import generate


def run() -> dict:
    print("=== Experiment: SHACL & Governance Rule Validation ===\n")

    # Generate benchmark data
    print("[1/4] Generating benchmark data...")
    summary = generate()
    print(f"  {summary['total_flights']} flights, {summary['graph_triples']} triples\n")

    # Load benchmark graph
    graph_path = Path(__file__).resolve().parent.parent / "benchmarks" / "sample_graphs" / "benchmark_graph.ttl"
    g = Graph()
    bind_namespaces(g)
    g.parse(str(graph_path), format="turtle")

    # SHACL validation
    print("[2/4] Running SHACL validation...")
    validator = ShaclValidator()
    report = validator.validate(g)
    print(f"  Conforms: {report.conforms}")
    print(f"  Violations found: {len(report.violations)}")
    for v in report.violations[:10]:
        print(f"    - {v.focus_node}: {v.message}")

    # Governance rules
    print("\n[3/4] Running governance rules...")
    gov_results = GovernanceRuleEngine.run_all(g)
    for r in gov_results:
        print(f"  [{r.rule_id}] {r.rule_label}: {len(r.affected_assets)} affected")
        print(f"    {r.explanation}")

    # Summary
    print("\n[4/4] Summary")
    result = {
        "shacl_conforms": report.conforms,
        "shacl_violations": len(report.violations),
        "governance_rules_executed": len(gov_results),
        "governance_affected_total": sum(len(r.affected_assets) for r in gov_results),
        "benchmark_flights": summary["total_flights"],
        "graph_triples": summary["graph_triples"],
    }
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    run()

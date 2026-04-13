"""Execute SPARQL competency and analytical queries against the benchmark graph.

Loads the benchmark RDF graph and runs all ``.rq`` query files, printing
results in a tabular format.

Usage::

    python -m SkyRwa.experiments.run_queries
"""

from __future__ import annotations

import sys
from pathlib import Path

_pkg = Path(__file__).resolve().parent.parent.parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from SkyRwa.rdf.graph_store import GraphStore
from SkyRwa.benchmarks.generate_benchmark import generate


def run() -> dict:
    print("=== SPARQL Query Runner ===\n")

    print("[1/3] Generating benchmark data...")
    generate()

    graph_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks" / "sample_graphs" / "benchmark_graph.ttl"
    )
    store = GraphStore()
    store.load_file(graph_path)
    store.load_ontology()
    print(f"  Loaded {len(store)} triples\n")

    base = Path(__file__).resolve().parent.parent / "queries"
    results: dict = {}

    for category in ["competency", "analytical"]:
        cat_dir = base / category
        if not cat_dir.exists():
            continue
        print(f"--- {category.upper()} QUERIES ---")
        for rq_file in sorted(cat_dir.glob("*.rq")):
            name = rq_file.stem
            print(f"\n  [{name}]")
            try:
                rows = store.query_file(rq_file)
                print(f"  Results: {len(rows)} rows")
                for row in rows[:5]:
                    formatted = {k: str(v)[:60] for k, v in row.items()}
                    print(f"    {formatted}")
                if len(rows) > 5:
                    print(f"    ... ({len(rows) - 5} more)")
                results[name] = len(rows)
            except Exception as e:
                print(f"  ERROR: {e}")
                results[name] = f"error: {e}"

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    run()

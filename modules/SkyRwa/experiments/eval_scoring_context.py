"""Experiment: cost of scoring-context materialization (H2 trade-off).

Quantifies what it costs to move the threshold and mission-state checks
(V3, V4, V6) from the procedural rule layer into the declarative SHACL
contract:

* graph size: triples per flight with and without materialized
  skyrwa:ScoringContext nodes;
* validation time: pySHACL over the baseline contract (shapes/) on the
  plain graph vs. the extended contract (shapes/ + shapes/extended/) on
  the context-materialized graph.

Coverage itself is measured by eval_ablation.py (SHACL+ctx column).

Usage::

    python -m SkyRwa.experiments.eval_scoring_context

Output: ISWC2026/outputs/scoring_context.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

_pkg = Path(__file__).resolve().parent.parent.parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from rdflib import Graph

from SkyRwa.ingest import FlightIngestor
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.rights import GovernanceEngine
from SkyRwa.valuation import RuleBasedValuationEngine
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.semantic_rules.validation_runner import ShaclValidator
from SkyRwa.experiments.eval_shacl_engines import _make_record

SIZES = [5, 20, 100, 500, 1000]
REPEATS = 3
OUTPUT = Path(__file__).resolve().parent.parent / "ISWC2026" / "outputs" / "scoring_context.json"


def _timed_validation(validator: ShaclValidator, g: Graph) -> tuple[float, bool]:
    times = []
    report = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        report = validator.validate(g)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), report.conforms


def run() -> dict:
    print("=== Experiment: scoring-context materialization cost ===\n")

    baseline_validator = ShaclValidator()
    extended_validator = ShaclValidator(include_extended=True)

    ingestor = FlightIngestor()
    eb = EvidenceBuilder()
    gov = GovernanceEngine()
    val = RuleBasedValuationEngine()

    results: dict = {
        "sizes": SIZES,
        "repeats": REPEATS,
        "note": (
            "baseline = shapes/ on plain graph; extended = shapes/ + "
            "shapes/extended/ on graph with materialized "
            "skyrwa:ScoringContext. pySHACL advanced=True; shape loading "
            "excluded; median of REPEATS runs."
        ),
        "runs": {},
    }

    for n in SIZES:
        units = []
        for i in range(n):
            rec = _make_record(i)
            u = ingestor.ingest(rec)
            u = eb.build(u, rec)
            gov.govern(u)
            val.evaluate(u)
            units.append(u)

        g_plain = Graph(); bind_namespaces(g_plain)
        mapper_plain = SkyRwaMapper(g_plain)
        for u in units:
            mapper_plain.map_asset_unit(u)

        g_ctx = Graph(); bind_namespaces(g_ctx)
        mapper_ctx = SkyRwaMapper(g_ctx, materialize_scoring_context=True)
        for u in units:
            mapper_ctx.map_asset_unit(u)

        base_ms, base_conforms = _timed_validation(baseline_validator, g_plain)
        ext_ms, ext_conforms = _timed_validation(extended_validator, g_ctx)

        t_plain, t_ctx = len(g_plain), len(g_ctx)
        print(f"--- N = {n} flights ---")
        print(f"  Triples:    {t_plain} -> {t_ctx} "
              f"(+{t_ctx - t_plain}, +{(t_ctx - t_plain) / n:.1f}/flight, "
              f"+{(t_ctx / t_plain - 1) * 100:.1f}%)")
        print(f"  Validation: {base_ms:.2f} ms (baseline) -> {ext_ms:.2f} ms "
              f"(extended, {'+' if ext_ms >= base_ms else ''}"
              f"{(ext_ms / base_ms - 1) * 100:.1f}%); "
              f"conforms: {base_conforms}/{ext_conforms}\n")

        results["runs"][f"n{n}"] = {
            "triples_plain": t_plain,
            "triples_ctx": t_ctx,
            "triples_added": t_ctx - t_plain,
            "triples_added_per_flight": round((t_ctx - t_plain) / n, 1),
            "graph_growth_pct": round((t_ctx / t_plain - 1) * 100, 1),
            "baseline_validation_ms": round(base_ms, 2),
            "extended_validation_ms": round(ext_ms, 2),
            "validation_delta_pct": round((ext_ms / base_ms - 1) * 100, 1),
            "baseline_conforms": base_conforms,
            "extended_conforms": ext_conforms,
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT}")
    return results


if __name__ == "__main__":
    run()

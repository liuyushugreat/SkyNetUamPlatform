"""Experiment: verify all 12 competency questions against an enriched
audit graph.

The pre-generated benchmark graph only covers the upstream tiers
(evidence, candidates, missions).  This script rebuilds the 105-flight
benchmark (seed 42) and enriches it with the downstream tiers so that
every CQ has a non-vacuous target:

* governed data products (CandidateAggregator + ProductBuilder),
* usage events and settlement records (Ledger + RevenueSplitter),
  including deliberately injected *post-settlement* usage for CQ9,
* governance decisions with ``prov:endedAtTime`` (GovernanceRuleEngine).

For each CQ it reports the number of SPARQL triple patterns (counted
from the parsed query algebra, so FILTER (NOT) EXISTS patterns are
included), the result count, and whether that count matches an expected
value computed independently from the Python domain objects.

Usage::

    python -m SkyRwa.experiments.eval_competency
"""

from __future__ import annotations

import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_pkg = Path(__file__).resolve().parent.parent.parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

from SkyRwa.benchmarks.generate_benchmark import SCENARIOS, _build_record
from SkyRwa.ingest import FlightIngestor
from SkyRwa.models.enums import AssetClass, UsageType
from SkyRwa.models.settlement import SettlementRule, SplitEntry
from SkyRwa.productization import CandidateAggregator, ProductBuilder
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.provenance.signing import Ed25519Signer
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.rights import GovernanceEngine
from SkyRwa.semantic_rules.governance_rules import GovernanceRuleEngine
from SkyRwa.settlement import Ledger
from SkyRwa.valuation import RuleBasedValuationEngine

QUERIES_DIR = Path(__file__).resolve().parent.parent / "queries" / "competency"
OUTPUT = (Path(__file__).resolve().parent.parent / "reproduction" / "outputs"
          / "competency.json")

SEED = 42
# CQ10 incident-investigation window (must match cq_10_incident_window.rq)
WINDOW_START = datetime(2026, 1, 17, 0, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 1, 19, 0, 0, 0, tzinfo=UTC)
# Historical usage timestamp base (before any settlement is recorded)
USAGE_BASE = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
N_POST_SETTLEMENT = 5  # assets that receive one post-settlement usage event


def _count_triple_patterns(query_text: str) -> int:
    """Count SPARQL triple patterns in the parsed query algebra,
    including patterns nested inside FILTER (NOT) EXISTS and OPTIONAL."""
    algebra = prepareQuery(query_text).algebra

    def walk(node) -> int:
        n = 0
        if hasattr(node, "keys"):  # CompValue is a Mapping
            for key in node.keys():
                val = node[key]
                if key == "triples" and isinstance(val, list):
                    n += len(val)
                else:
                    n += walk(val)
        elif isinstance(node, list):
            for item in node:
                n += walk(item)
        return n

    return walk(algebra)


def build_enriched_graph() -> tuple[Graph, dict]:
    """Run the 105-flight benchmark pipeline end to end (all six tiers)
    and return the enriched graph plus ground-truth expectations."""
    random.seed(SEED)

    ingestor = FlightIngestor()
    evidence_builder = EvidenceBuilder()
    governance = GovernanceEngine()
    valuation = RuleBasedValuationEngine()
    signer = Ed25519Signer.generate_keypair("competency-signer")
    base_time = datetime(2026, 1, 15, 8, 0, 0, tzinfo=UTC)

    units = []
    for scenario in SCENARIOS:
        record = _build_record(scenario, base_time)
        unit = ingestor.ingest(record)
        unit = evidence_builder.build(unit, record)
        signer.sign_evidence(unit.evidence)
        governance.govern(unit)
        valuation.evaluate(unit)
        units.append(unit)

    # ── Tier 4: governed data products ──────────────────────────────────
    aggregator = CandidateAggregator(min_count=3)
    builder = ProductBuilder()
    products = [builder.build(g) for g in aggregator.group(units).values()]
    promoted_ids = {aid for p in products for aid in p.source_asset_ids}

    # ── Tier 5/6: usage events + settlement records ────────────────────
    # Settle the candidates of the largest product; the first
    # N_POST_SETTLEMENT of them additionally get one usage event *after*
    # settlement (license-drift scenario for CQ9).
    largest = max(products, key=lambda p: len(p.source_asset_ids))
    settled_units = [u for u in units
                     if u.asset_unit_id in set(largest.source_asset_ids)]
    ledger = Ledger()
    settlement_records = []
    post_settlement_usage = 0
    for i, unit in enumerate(settled_units):
        participants = [
            SplitEntry(party_id=p.party_id, role=p.role, share_pct=p.share_pct)
            for p in (unit.rights_profile.revenue_split
                      if unit.rights_profile else [])
        ]
        unit.settlement_rule = SettlementRule(participants=participants)

        for j in range(2):
            log = ledger.record_usage(
                unit, UsageType.API_CALL, f"consumer-{i % 3}",
                gross_amount=10.0 + i + j,
            )
            log.timestamp = USAGE_BASE + timedelta(hours=6 * i + j)

        rec = ledger.settle_all(unit.asset_unit_id)
        settlement_records.append(rec)

        if i < N_POST_SETTLEMENT:
            log = ledger.record_usage(
                unit, UsageType.API_CALL, "consumer-late",
                gross_amount=5.0,
            )
            log.timestamp = rec.settled_at + timedelta(days=3)
            post_settlement_usage += 1

    # ── Map everything to RDF ───────────────────────────────────────────
    graph = Graph()
    bind_namespaces(graph)
    mapper = SkyRwaMapper(graph)
    for unit in units:
        mapper.map_asset_unit(unit)  # also maps unit.revenue_log entries
    for product in products:
        mapper.map_product(product)
    for rec in settlement_records:
        mapper.map_settlement_record(rec)

    # ── Governance decisions (with prov:endedAtTime) ────────────────────
    rule_results = GovernanceRuleEngine.run_all(graph)
    GovernanceRuleEngine.inject_decisions(graph, rule_results)

    # ── Ground truth from the domain objects ────────────────────────────
    tradable = [u for u in units
                if u.rights_profile and u.rights_profile.tradable]
    route_products = [p for p in products
                      if p.product_category ==
                      AssetClass.ROUTE_OPTIMIZATION_SAMPLE]
    party_roles = {(s.party_id, s.role)
                   for rec in settlement_records
                   for s in rec.participant_totals}
    settled_total = round(sum(
        s.amount for rec in settlement_records for s in rec.participant_totals
    ), 2)
    uav_counts = {
        p.product_id: len({
            u.uav_id for u in units
            if u.asset_unit_id in set(p.source_asset_ids)
        })
        for p in products
    }

    expected = {
        "cq_01": len(tradable),
        "cq_02": len([u for u in units if u.rights_profile
                      and u.rights_profile.desensitization_required]),
        "cq_03": sum(len(p.source_asset_ids) for p in route_products),
        "cq_04": len(party_roles),
        "cq_05": 0,  # governance always attaches a rights profile
        "cq_06": len([p for p in products
                      if len(p.source_asset_ids) > 1]),
        "cq_07": 0,  # every evidence package feeds exactly one candidate
        "cq_08": len(units) - len(promoted_ids),
        "cq_09": post_settlement_usage,
        "cq_10": len([u for u in units if u.start_time
                      and WINDOW_START <= u.start_time < WINDOW_END]),
        "cq_11": len(products),
        "cq_12": len([c for c in uav_counts.values() if c > 3]),
    }
    stats = {
        "flights": len(units),
        "tradable": len(tradable),
        "products": len(products),
        "promoted_candidates": len(promoted_ids),
        "settled_assets": len(settlement_records),
        "settled_total": settled_total,
        "usage_events": len(ledger.entries),
        "post_settlement_usage": post_settlement_usage,
        "decisions": sum(len(r.affected_assets) for r in rule_results),
        "triples": len(graph),
        "expected": expected,
    }
    return graph, stats


def run() -> dict:
    print("=== Experiment: Competency-question verification (12 CQs) ===\n")

    print("[1/2] Building enriched audit graph (105 flights, all tiers)...")
    graph, stats = build_enriched_graph()
    print(f"  {stats['triples']} triples | {stats['products']} products | "
          f"{stats['settled_assets']} settled assets | "
          f"{stats['usage_events']} usage events | "
          f"{stats['decisions']} decisions\n")

    print("[2/2] Running competency questions...")
    print(f"  {'CQ':<6} {'patterns':>8} {'rows':>6} {'expected':>9}  verdict")
    print("  " + "-" * 50)

    results = {"stats": {k: v for k, v in stats.items() if k != "expected"},
               "cqs": []}
    all_ok = True
    for rq in sorted(QUERIES_DIR.glob("cq_*.rq")):
        key = rq.stem[:5]  # e.g. "cq_07"
        text = rq.read_text(encoding="utf-8")
        rows = len(list(graph.query(text)))
        patterns = _count_triple_patterns(text)
        expected = stats["expected"][key]
        ok = rows == expected
        all_ok &= ok
        print(f"  {key:<6} {patterns:>8} {rows:>6} {expected:>9}  "
              f"{'OK' if ok else 'MISMATCH'}")
        results["cqs"].append({
            "cq": key, "file": rq.name, "patterns": patterns,
            "rows": rows, "expected": expected, "correct": ok,
        })

    print("  " + "-" * 50)
    print(f"  All correct: {all_ok}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {OUTPUT}")
    return results


if __name__ == "__main__":
    # Non-zero exit on any CQ mismatch so CI fails loudly.
    sys.exit(0 if all(c["correct"] for c in run()["cqs"]) else 1)

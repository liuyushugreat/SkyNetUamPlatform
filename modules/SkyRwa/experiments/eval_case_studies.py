"""Experiment: Case studies for the ISWC paper.

Produces 3 detailed cases that can be directly included in the paper's
evaluation section:

1. Clean promotable case — a group of flights that pass governance and
   get aggregated into a GovernedDataProduct.
2. Governance failure case — flights with violations that are correctly
   blocked from trading.
3. Multi-flight productization case — full lifecycle from evidence
   through product to valuation explanation.

Usage::

    python -m SkyRwa.experiments.eval_case_studies
"""

from __future__ import annotations

import json
import sys
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
from SkyRwa.settlement import Ledger, RevenueSplitter
from SkyRwa.productization import CandidateAggregator, ProductBuilder, ProductCatalogue
from SkyRwa.valuation.product_valuation import ProductValuationEngine
from SkyRwa.semantic_rules.explanation_rules import ExplanationBuilder
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.models.enums import AssetClass


def _make_records(prefix: str, n: int, **overrides) -> list:
    base = datetime(2026, 2, 1, 9, 0, 0, tzinfo=UTC)
    records = []
    for i in range(n):
        start = base + timedelta(hours=i * 3)
        defaults = dict(
            flight_id=f"{prefix}-{i:03d}",
            uav_id=f"UAV-{prefix[-1]}{i%3+1}",
            mission_id=f"MSN-{prefix}-{i}",
            operator_id="OP-CASE",
            start_time=start,
            end_time=start + timedelta(minutes=25),
            mission_type="route_survey",
            mission_completed=True,
            completion_pct=100.0,
            telemetry_points=1200,
            avg_altitude_m=150.0,
            weather_condition="clear",
            wind_speed_mps=3.0,
            visibility_km=15.0,
        )
        defaults.update(overrides)
        records.append(FlightIngestRecord(**defaults))
    return records


def run() -> dict:
    print("=== Case Studies for ISWC Paper ===\n")
    results = {}

    ingestor = FlightIngestor()
    eb = EvidenceBuilder()
    gov = GovernanceEngine()
    val = RuleBasedValuationEngine()
    signer = Ed25519Signer.generate_keypair("case-study-signer")
    explainer = ExplanationBuilder()

    # ── Case 1: Clean promotable ──────────────────────────────────────────
    print("== Case 1: Clean Promotable Group ==")
    clean_records = _make_records("FLT-CS1", 5)
    clean_units = []
    for rec in clean_records:
        u = ingestor.ingest(rec)
        u = eb.build(u, rec)
        signer.sign_evidence(u.evidence)
        gov.govern(u)
        val.evaluate(u)
        clean_units.append(u)

    for u in clean_units:
        exp = explainer.explain_governance(u)
        print(f"  {u.flight_id}: tradable={u.rights_profile.tradable if u.rights_profile else 'N/A'}, "
              f"quality={u.data_quality_score:.2f}, compliance={u.compliance_score:.2f}")
        print(f"    {exp.conclusion}")

    agg = CandidateAggregator(min_count=3)
    groups = agg.group(clean_units)
    builder = ProductBuilder()
    prod_val = ProductValuationEngine()
    catalogue = ProductCatalogue()

    for cls, group in groups.items():
        product = builder.build(group)
        pval = prod_val.valuate(product)
        catalogue.register(product)
        print(f"\n  Product: {product.product_id[:12]}... ({cls.value})")
        print(f"    Sources: {group.count}, Value: {pval.final_value:.2f}")
        print(f"    {pval.promotion_readiness}")

    results["case1_flights"] = len(clean_units)
    results["case1_products"] = len(catalogue)
    results["case1_all_tradable"] = all(
        u.rights_profile.tradable for u in clean_units if u.rights_profile
    )

    # ── Case 2: Governance failure ────────────────────────────────────────
    print("\n== Case 2: Governance Failure ==")
    fail_records = _make_records(
        "FLT-CS2", 3,
        mission_completed=False,
        completion_pct=40.0,
        violations=["altitude_exceedance", "nfz_incursion"],
        anomalies=["sensor_failure"],
        no_fly_zone_incursions=2,
    )
    fail_units = []
    for rec in fail_records:
        u = ingestor.ingest(rec)
        u = eb.build(u, rec)
        gov.govern(u)
        val.evaluate(u)
        fail_units.append(u)

    for u in fail_units:
        exp = explainer.explain_governance(u)
        print(f"  {u.flight_id}: tradable={u.rights_profile.tradable if u.rights_profile else 'N/A'}, "
              f"compliance={u.compliance_score:.2f}, risk={u.risk_score:.2f}")
        print(f"    {exp.conclusion}")

    results["case2_flights"] = len(fail_units)
    results["case2_all_non_tradable"] = all(
        not u.rights_profile.tradable for u in fail_units if u.rights_profile
    )

    # ── Case 3: Multi-flight productization ───────────────────────────────
    print("\n== Case 3: Multi-flight Productization Lifecycle ==")
    multi_records = _make_records(
        "FLT-CS3", 6,
        mission_type="weather_monitoring",
            weather_condition="rainy",
            wind_speed_mps=10.0,
            visibility_km=4.0,
            telemetry_points=900,
    )
    multi_units = []
    g = Graph()
    bind_namespaces(g)
    mapper = SkyRwaMapper(g)

    for rec in multi_records:
        u = ingestor.ingest(rec)
        u = eb.build(u, rec)
        signer.sign_evidence(u.evidence)
        gov.govern(u)
        val.evaluate(u)
        u.asset_class = AssetClass.WEATHER_OPERATION_SAMPLE
        multi_units.append(u)
        mapper.map_asset_unit(u)

    groups = agg.group(multi_units)
    for cls, group in groups.items():
        product = builder.build(group)
        pval = prod_val.valuate(product)
        catalogue.register(product)

        print(f"  Product: {product.product_id[:12]}... ({cls.value})")
        print(f"    Sources: {group.count}")
        print(f"    Avg quality: {group.avg_quality:.3f}")
        print(f"    Product value: {pval.final_value:.2f}")
        print(f"    Promotion readiness: {pval.promotion_readiness}")
        print(f"    Lineage: {product.lineage_note}")

        val_exp = explainer.explain_valuation(multi_units[0])
        print(f"\n    Valuation explanation for first source:")
        print(f"      {val_exp.conclusion}")
        for f in val_exp.factors[:3]:
            print(f"      [{f.impact}] {f.dimension}: {f.score:.2f}")

    results["case3_flights"] = len(multi_units)
    results["case3_products"] = len(groups)
    results["case3_graph_triples"] = len(g)

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    run()

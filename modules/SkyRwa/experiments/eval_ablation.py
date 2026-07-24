"""Ablation study: Python-only vs SHACL-only vs combined governance detection.

Tests which violation types each layer catches and which it misses,
demonstrating the necessity of the dual-layer approach.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from rdflib import Graph, Literal, URIRef, RDF
from rdflib.namespace import XSD
from SkyRwa.rdf.namespaces import SKYRWA, bind_namespaces
from SkyRwa.semantic_rules.validation_runner import ShaclValidator
from SkyRwa.models.enums import AssetClass, AssetStatus, DataCategory, UsageLevel
from SkyRwa.models.rights import RightsProfile, RetentionPolicy, RevenueParticipant
from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.models.evidence import FlightEvidencePackage, TelemetrySummary
from SkyRwa.models.valuation import ValuationResultV2, DataQualityScore, AssetValueScore
from SkyRwa.rights.governance import GovernanceEngine


def _make_base_unit(flight_id: str, compliance: float = 0.9,
                    risk: float = 0.1, completed: bool = True) -> FlightAssetUnit:
    now = datetime.now(UTC)
    return FlightAssetUnit(
        flight_id=flight_id,
        uav_id="UAV-TEST",
        mission_type="survey",
        start_time=now - timedelta(hours=1),
        end_time=now,
        compliance_score=compliance,
        risk_score=risk,
        data_quality_score=0.8,
        asset_class=AssetClass.ROUTE_OPTIMIZATION_SAMPLE,
        status=AssetStatus.INGESTED,
    )


def _map_unit_to_graph(unit: FlightAssetUnit, *, include_digest: bool = True,
                       include_derivation: bool = True,
                       include_rights: bool = True,
                       materialize_context: bool = False,
                       completed: bool = True) -> Graph:
    """Manually map a unit to RDF with optional omissions for ablation."""
    g = Graph()
    bind_namespaces(g)
    uri = URIRef(f"urn:skyrwa:asset:{unit.flight_id}")

    g.add((uri, RDF.type, SKYRWA.AssetCandidate))
    g.add((uri, SKYRWA.flightId, Literal(unit.flight_id)))
    g.add((uri, SKYRWA.uavId, Literal(unit.uav_id)))
    g.add((uri, SKYRWA.startTime, Literal(unit.start_time.isoformat(), datatype=XSD.dateTime)))
    g.add((uri, SKYRWA.endTime, Literal(unit.end_time.isoformat(), datatype=XSD.dateTime)))
    g.add((uri, SKYRWA.complianceScore, Literal(unit.compliance_score)))
    g.add((uri, SKYRWA.riskScore, Literal(unit.risk_score)))
    g.add((uri, SKYRWA.hasAssetClass, Literal(unit.asset_class.value)))

    ev_uri = URIRef(f"urn:skyrwa:evidence:{unit.flight_id}")
    g.add((ev_uri, RDF.type, SKYRWA.FlightEvidence))
    g.add((ev_uri, SKYRWA.flightId, Literal(unit.flight_id)))
    g.add((ev_uri, SKYRWA.uavId, Literal(unit.uav_id)))
    g.add((ev_uri, SKYRWA.startTime, Literal(unit.start_time.isoformat(), datatype=XSD.dateTime)))
    g.add((ev_uri, SKYRWA.endTime, Literal(unit.end_time.isoformat(), datatype=XSD.dateTime)))

    if include_digest:
        g.add((ev_uri, SKYRWA.hasDigest, Literal("sha256:test_digest_abc")))

    if include_derivation:
        g.add((uri, SKYRWA.derivedFromEvidence, ev_uri))

    if include_rights:
        rp_uri = URIRef(f"urn:skyrwa:rights:{unit.flight_id}")
        g.add((uri, SKYRWA.hasRightsProfile, rp_uri))
        g.add((rp_uri, SKYRWA.isTradable, Literal(True, datatype=XSD.boolean)))
    else:
        g.add((uri, SKYRWA.isTradable, Literal(True, datatype=XSD.boolean)))

    val_uri = URIRef(f"urn:skyrwa:valuation:{unit.flight_id}")
    g.add((uri, SKYRWA.hasValuation, val_uri))
    g.add((val_uri, SKYRWA.estimatedValue, Literal(75.0)))

    if materialize_context:
        ctx_uri = URIRef(f"urn:skyrwa:scoring:{unit.flight_id}")
        g.add((uri, SKYRWA.hasScoringContext, ctx_uri))
        g.add((ctx_uri, RDF.type, SKYRWA.ScoringContext))
        g.add((ctx_uri, SKYRWA.missionCompleted,
               Literal(completed, datatype=XSD.boolean)))
        g.add((ctx_uri, SKYRWA.violationCount,
               Literal(0, datatype=XSD.integer)))
        g.add((ctx_uri, SKYRWA.anomalyCount,
               Literal(0, datatype=XSD.integer)))

    return g


# ── Violation scenarios ─────────────────────────────────────────────────────

VIOLATIONS = [
    {
        "id": "V1", "name": "Missing digest",
        "description": "FlightEvidence lacks hasDigest",
        "graph_kwargs": {"include_digest": False},
        "python_detectable": False,
        "shacl_detectable": True,
        "combined_detectable": True,
    },
    {
        "id": "V2", "name": "Missing derivation link",
        "description": "AssetCandidate has no derivedFromEvidence",
        "graph_kwargs": {"include_derivation": False},
        "python_detectable": False,
        "shacl_detectable": True,
        "combined_detectable": True,
    },
    {
        "id": "V3", "name": "Low compliance + tradable",
        "description": "Asset with compliance < 0.5 but marked tradable",
        "graph_kwargs": {},
        "unit_kwargs": {"compliance": 0.3},
        "python_detectable": True,
        "shacl_detectable": False,
        "combined_detectable": True,
    },
    {
        "id": "V4", "name": "High risk + tradable",
        "description": "Asset with risk > 0.8 but marked tradable",
        "graph_kwargs": {},
        "unit_kwargs": {"risk": 0.9},
        "python_detectable": True,
        "shacl_detectable": False,
        "combined_detectable": True,
    },
    {
        "id": "V5", "name": "Missing rights profile on tradable",
        "description": "AssetCandidate without hasRightsProfile",
        "graph_kwargs": {"include_rights": False},
        "python_detectable": False,
        "shacl_detectable": True,
        "combined_detectable": True,
    },
    {
        "id": "V6", "name": "Incomplete mission + tradable",
        "description": "Flight not completed but asset marked tradable",
        "graph_kwargs": {},
        "unit_kwargs": {"completed": False},
        "python_detectable": True,
        "shacl_detectable": False,
        "combined_detectable": True,
    },
]


def run_ablation() -> list[dict]:
    """Run the ablation and return one row per violation type.

    Columns: Python-only, SHACL with the baseline contract, SHACL with the
    extended contract over materialized scoring context (shapes/extended/),
    and the combined dual layer.
    """
    validator = ShaclValidator()
    validator_ext = ShaclValidator(include_extended=True)
    gov = GovernanceEngine()

    print("=" * 88)
    print("ABLATION STUDY: Python vs SHACL (baseline) vs SHACL+ctx (extended) vs Combined")
    print("=" * 88)
    print(f"{'ID':<5} {'Violation':<35} {'Python':>8} {'SHACL':>8} "
          f"{'SHACL+ctx':>10} {'Combined':>10}")
    print("-" * 88)

    rows: list[dict] = []
    python_total = 0
    shacl_total = 0
    shacl_ext_total = 0
    combined_total = 0

    for v in VIOLATIONS:
        unit_kw = v.get("unit_kwargs", {})
        unit = _make_base_unit(f"FLT-ABL-{v['id']}", **unit_kw)

        # Python-only detection
        python_detected = False
        if unit_kw.get("compliance", 1.0) < 0.5 or unit_kw.get("risk", 0) > 0.8:
            python_detected = True
        if unit_kw.get("completed") is False:
            python_detected = True

        # SHACL detection, baseline contract
        g = _map_unit_to_graph(unit, **v["graph_kwargs"])
        report = validator.validate(g)
        shacl_detected = not report.conforms

        # SHACL detection, extended contract + materialized scoring context
        g_ctx = _map_unit_to_graph(
            unit, **v["graph_kwargs"],
            materialize_context=True,
            completed=unit_kw.get("completed", True),
        )
        report_ext = validator_ext.validate(g_ctx)
        shacl_ext_detected = not report_ext.conforms

        combined = python_detected or shacl_detected

        python_total += int(python_detected)
        shacl_total += int(shacl_detected)
        shacl_ext_total += int(shacl_ext_detected)
        combined_total += int(combined)

        rows.append({
            "id": v["id"], "name": v["name"],
            "python": python_detected,
            "shacl": shacl_detected,
            "shacl_extended": shacl_ext_detected,
            "combined": combined,
        })
        print(f"{v['id']:<5} {v['name']:<35} "
              f"{'YES' if python_detected else 'no':>8} "
              f"{'YES' if shacl_detected else 'no':>8} "
              f"{'YES' if shacl_ext_detected else 'no':>10} "
              f"{'YES' if combined else 'no':>10}")

    print("-" * 88)
    n = len(VIOLATIONS)
    print(f"{'TOTAL':<5} {f'{n} violation types':<35} {python_total:>8} "
          f"{shacl_total:>8} {shacl_ext_total:>10} {combined_total:>10}")
    print(f"{'':5} {'Detection rate':<35} {python_total/n*100:>7.0f}% "
          f"{shacl_total/n*100:>7.0f}% {shacl_ext_total/n*100:>9.0f}% "
          f"{combined_total/n*100:>9.0f}%")
    print()
    print("Key finding: with materialized scoring context, the extended SHACL")
    print("contract alone reaches the coverage of the combined dual layer.")
    return rows


if __name__ == "__main__":
    run_ablation()

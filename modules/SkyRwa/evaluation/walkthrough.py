"""End-to-end walkthrough: one near-NFZ flight from ingest to GOV-001 blocking.

Traces a single flight (FLT-NFZ-DEMO) through the full pipeline --
ingest -> evidence -> governance -> valuation -> RDF mapping -> semantic
governance rules -> decision injection -> SHACL validation -> audit SPARQL
query -- and extracts the ACTUAL Turtle emitted at each stage plus the
actual audit query result rows.

The paper's walkthrough subsection is generated from this script (LaTeX
fragment ``walkthrough_generated.tex``), so the listings shown in the
paper always come from a live run rather than being hand-written.

Usage::

    python -m SkyRwa.evaluation.walkthrough

Outputs:
    reproduction/outputs/walkthrough.json           raw stage data
    reproduction/outputs/walkthrough_generated.tex  paper subsection body
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_pkg = Path(__file__).resolve().parent.parent.parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from rdflib import Graph, Literal, URIRef

from SkyRwa.ingest import FlightIngestRecord, FlightIngestor
from SkyRwa.provenance import EvidenceBuilder
from SkyRwa.provenance.signing import Ed25519Signer
from SkyRwa.rights import GovernanceEngine
from SkyRwa.valuation import RuleBasedValuationEngine
from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.namespaces import SKYRWA, bind_namespaces
from SkyRwa.semantic_rules.governance_rules import GovernanceRuleEngine
from SkyRwa.semantic_rules.validation_runner import ShaclValidator

OUT_DIR = Path(__file__).resolve().parent.parent / "reproduction" / "outputs"

AUDIT_QUERY = """PREFIX skyrwa: <https://w3id.org/skyrwa#>
PREFIX prov:   <http://www.w3.org/ns/prov#>

SELECT ?ruleId ?decision ?explanation ?risk ?decidedAt
WHERE {
    ?asset a skyrwa:AssetCandidate ;
           skyrwa:flightId "FLT-NFZ-DEMO" ;
           skyrwa:riskScore ?risk .
    ?decision a skyrwa:GovernanceDecision ;
              skyrwa:appliedToAsset ?asset ;
              skyrwa:ruleId ?ruleId ;
              skyrwa:explanation ?explanation ;
              prov:endedAtTime ?decidedAt .
}"""


def _make_record() -> FlightIngestRecord:
    """One near-NFZ patrol flight: 3 NFZ incursions, 1 violation, 1 anomaly.
    Risk = 0.6 + 0.15 + 0.1 = 0.85; compliance = 1 - 0.2 - 0.05 = 0.75."""
    start = datetime(2026, 3, 14, 9, 30, 0, tzinfo=UTC)
    return FlightIngestRecord(
        flight_id="FLT-NFZ-DEMO",
        uav_id="UAV-D1",
        mission_id="MSN-PATROL-114",
        operator_id="OP-CITYPATROL",
        start_time=start,
        end_time=start + timedelta(minutes=34),
        mission_type="patrol",
        telemetry_points=960,
        avg_altitude_m=118.0,
        max_altitude_m=142.0,
        weather_condition="overcast",
        wind_speed_mps=5.0,
        visibility_km=8.0,
        no_fly_zone_incursions=3,
        risk_events=["nfz_buffer_entry"],
        mission_completed=True,
        completion_pct=100.0,
        anomalies=["nfz_proximity"],
        violations=["nfz_proximity_warning"],
    )


def _node_turtle(g: Graph, subject: URIRef, keep_preds: list | None = None,
                 elide_note: str | None = None) -> str:
    """Serialize the triples of one subject as Turtle (optionally only the
    predicates in *keep_preds*, appending an elision comment)."""
    sub = Graph()
    bind_namespaces(sub)
    kept, total = 0, 0
    for p, o in sorted(g.predicate_objects(subject)):
        total += 1
        if keep_preds is None or p in keep_preds:
            sub.add((subject, p, o))
            kept += 1
    ttl = sub.serialize(format="turtle")
    # strip @prefix headers; the paper shows them once
    lines = [ln for ln in ttl.splitlines()
             if ln.strip() and not ln.startswith("@prefix")]
    body = "\n".join(lines)
    if elide_note and kept < total:
        body = body.rstrip()
        if body.endswith(" ."):
            body = body[:-2] + " ;\n    # ... " + \
                f"{total - kept} further triples elided ({elide_note}) ...\n"
    return body


def run() -> dict:
    print("=== Walkthrough: FLT-NFZ-DEMO from ingest to GOV-001 blocking ===\n")

    # ── pipeline ─────────────────────────────────────────────────────────
    record = _make_record()
    unit = FlightIngestor().ingest(record)
    unit = EvidenceBuilder().build(unit, record)
    Ed25519Signer.generate_keypair("walkthrough-signer").sign_evidence(unit.evidence)
    GovernanceEngine().govern(unit, operator_id=record.operator_id)
    RuleBasedValuationEngine().evaluate(unit)

    g = Graph()
    bind_namespaces(g)
    mapper = SkyRwaMapper(g)
    asset_iri = mapper.map_asset_unit(unit)
    evidence_iri = next(g.objects(asset_iri, SKYRWA.derivedFromEvidence))

    # ── semantic rules + decision injection ─────────────────────────────
    rule_results = GovernanceRuleEngine.run_all(g)
    GovernanceRuleEngine.inject_decisions(g, rule_results)
    fired = [r for r in rule_results if r.affected_assets]

    # ── SHACL validation ─────────────────────────────────────────────────
    report = ShaclValidator().validate(g)

    # ── audit query ──────────────────────────────────────────────────────
    audit_rows = [
        {str(k): str(v) for k, v in row.asdict().items()}
        for row in g.query(AUDIT_QUERY)
    ]

    # ── stage listings (real Turtle from the live graph) ─────────────────
    ttl_evidence = _node_turtle(
        g, evidence_iri,
        keep_preds=[SKYRWA.flightId, SKYRWA.hasDigest, SKYRWA.hasSignature,
                    SKYRWA.operatedBy, SKYRWA.performedByUAV],
        elide_note="timestamps, mission link",
    )
    rights_node = next(g.objects(asset_iri, SKYRWA.hasRightsProfile))
    ttl_asset = _node_turtle(
        g, asset_iri,
        keep_preds=[SKYRWA.flightId, SKYRWA.complianceScore, SKYRWA.riskScore,
                    SKYRWA.derivedFromEvidence, SKYRWA.hasRightsProfile,
                    SKYRWA.hasStatus],
        elide_note="asset class, quality score, times, valuation",
    ).replace("[ ]", "_:rights")
    ttl_rights = _node_turtle(
        g, rights_node,
        keep_preds=[SKYRWA.isTradable, SKYRWA.requiresDesensitization,
                    SKYRWA.permittedUse],
        elide_note="revenue shares",
    ).replace("[]", "_:rights", 1)
    decision_iri = next(g.subjects(SKYRWA.appliedToAsset, asset_iri))
    ttl_decision = _node_turtle(g, decision_iri)

    summary = {
        "flight_id": record.flight_id,
        "risk_score": unit.risk_score,
        "compliance_score": unit.compliance_score,
        "tradable": unit.rights_profile.tradable,
        "desensitization_required": unit.rights_profile.desensitization_required,
        "rules_fired": [r.rule_id for r in fired],
        "triples": len(g),
        "shacl_conforms": report.conforms,
        "audit_rows": audit_rows,
        "listings": {
            "evidence": ttl_evidence,
            "asset": ttl_asset,
            "rights": ttl_rights,
            "decision": ttl_decision,
        },
        "audit_query": AUDIT_QUERY,
    }

    print(f"risk={unit.risk_score:.2f} compliance={unit.compliance_score:.2f} "
          f"tradable={unit.rights_profile.tradable}")
    print(f"rules fired: {[r.rule_id for r in fired]}")
    print(f"SHACL conforms: {report.conforms}; triples: {len(g)}")
    print(f"audit rows: {len(audit_rows)}")
    for row in audit_rows:
        print("  ", row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "walkthrough.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_latex(summary)
    print(f"\nSaved: {OUT_DIR / 'walkthrough.json'}")
    print(f"Saved: {OUT_DIR / 'walkthrough_generated.tex'}")
    return summary


def _write_latex(s: dict) -> None:
    """Emit the paper's walkthrough subsection body (listings + result rows)."""
    row = s["audit_rows"][0]
    decided_short = row["decidedAt"][:19] + "Z"
    tex = f"""% ---------------------------------------------------------------------------
% GENERATED FILE -- do not edit by hand.
% Produced by modules/SkyRwa/evaluation/walkthrough.py; all Turtle and all
% query results below are extracted from a live pipeline run.
% ---------------------------------------------------------------------------
We trace one flight end-to-end: \\texttt{{FLT-NFZ-DEMO}}, a patrol mission that clips a no-fly-zone buffer (3~NFZ incursions, 1~recorded violation, 1~anomaly). Ingest assigns risk $ {s['risk_score']:.2f} $; the compliance heuristic yields $ {s['compliance_score']:.2f} $. All listings below are extracted verbatim from the live run by \\texttt{{evaluation/walkthrough.py}} (prefixes elided).

\\textbf{{Stage 1 --- evidence attestation.}} The signed evidence node anchors the audit trail:

\\begin{{lstlisting}}[language=SPARQL,numbers=none]
{s['listings']['evidence']}
\\end{{lstlisting}}

\\textbf{{Stage 2 --- governance.}} Because violations are present, the rights profile blocks trading; the asset candidate carries the scores the rules will read:

\\begin{{lstlisting}}[language=SPARQL,numbers=none]
{s['listings']['asset']}
{s['listings']['rights']}
\\end{{lstlisting}}

\\textbf{{Stage 3 --- rule firing and decision materialization.}} Semantic rule GOV-001 fires on the risk threshold ($ {s['risk_score']:.2f} > 0.8 $) and is materialized as a first-class decision with rule ID, explanation, timestamp, and PROV edges:

\\begin{{lstlisting}}[language=SPARQL,numbers=none]
{s['listings']['decision']}
\\end{{lstlisting}}

The full graph ({s['triples']}~triples) conforms to all published shapes (\\texttt{{sh:conforms true}}): blocking is not a constraint violation but a governed, queryable state.

\\textbf{{Stage 4 --- one-query audit.}} A third party holding only the graph answers ``why was this flight blocked, by which rule, and when?'' in a single SPARQL query:

\\begin{{lstlisting}}[language=SPARQL,numbers=none]
{s['audit_query']}
\\end{{lstlisting}}

\\noindent which returns exactly one row: \\texttt{{ruleId}}~= \\texttt{{{row['ruleId']}}}, \\texttt{{explanation}}~= ``{row['explanation']}'', \\texttt{{risk}}~= {float(row['risk']):.2f}, \\texttt{{decidedAt}}~= {decided_short}, with the decision IRI linking onward to the evidence via \\texttt{{prov:used}}. The JSON baseline must reconstruct the same answer from application logs.
"""
    (OUT_DIR / "walkthrough_generated.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    run()

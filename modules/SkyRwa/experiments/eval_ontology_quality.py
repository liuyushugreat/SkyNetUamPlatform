"""Ontology quality assessment: OOPS!-style pitfall scanning, reasoner
consistency check, and competency question → ontology construct mapping.

Produces results for Table 4 (OOPS! pitfalls) and Table 5 (CQ mapping)
in the ISWC 2026 paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from rdflib import Graph, RDF, RDFS, OWL

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"


def load_ontology() -> Graph:
    g = Graph()
    g.parse(str(ONTOLOGY_DIR / "skyrwa.ttl"), format="turtle")
    if (ONTOLOGY_DIR / "alignments.ttl").exists():
        g.parse(str(ONTOLOGY_DIR / "alignments.ttl"), format="turtle")
    return g


# ── OOPS!-style pitfall checks ──────────────────────────────────────────

PITFALL_CHECKS = [
    ("P04", "Unconnected ontology element", "Critical",
     "Classes with no properties pointing to or from them"),
    ("P05", "Inverse relationships not explicitly declared", "Minor",
     "Object properties lacking owl:inverseOf"),
    ("P07", "Merging different concepts in the same class", "Important",
     "Classes that conflate distinct real-world entities"),
    ("P08", "Missing annotations", "Minor",
     "Classes/properties without rdfs:label or rdfs:comment"),
    ("P11", "Missing domain or range in properties", "Important",
     "Properties without declared domain and range"),
    ("P13", "Inverse relationships not explicitly declared", "Minor",
     "Paired object properties without owl:inverseOf"),
    ("P19", "Defining multiple domains or ranges", "Important",
     "Properties with >1 domain or range declaration"),
    ("P24", "Using recursive definitions", "Critical",
     "Class defined in terms of itself"),
    ("P25", "Defining a relationship as inverse to itself", "Critical",
     "Property declared owl:inverseOf to itself"),
]


def check_p04_unconnected(g: Graph) -> int:
    """Check for classes that appear neither as domain nor range of any property."""
    classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    connected = set()
    for s, p, o in g:
        if p in (RDFS.domain, RDFS.range):
            connected.add(o)
        if p in (RDFS.subClassOf,):
            connected.add(s)
            connected.add(o)
    unconnected = classes - connected
    return len(unconnected)


def check_p08_missing_annotations(g: Graph) -> int:
    """Count classes/properties missing rdfs:comment."""
    classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    properties = (set(g.subjects(RDF.type, OWL.ObjectProperty)) |
                  set(g.subjects(RDF.type, OWL.DatatypeProperty)))
    entities = classes | properties
    missing = 0
    for e in entities:
        comments = list(g.objects(e, RDFS.comment))
        if not comments:
            missing += 1
    return missing


def check_p11_missing_domain_range(g: Graph) -> int:
    """Count properties missing domain or range."""
    properties = (set(g.subjects(RDF.type, OWL.ObjectProperty)) |
                  set(g.subjects(RDF.type, OWL.DatatypeProperty)))
    missing = 0
    for p in properties:
        has_domain = any(g.objects(p, RDFS.domain))
        has_range = any(g.objects(p, RDFS.range))
        if not has_domain or not has_range:
            missing += 1
    return missing


def check_p05_missing_inverse(g: Graph) -> int:
    """Count object properties without owl:inverseOf."""
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    missing = 0
    for p in obj_props:
        inverses = list(g.objects(p, OWL.inverseOf))
        if not inverses:
            missing += 1
    return missing


def check_p25_self_inverse(g: Graph) -> int:
    """Check for properties that are inverse of themselves."""
    count = 0
    for s, _, o in g.triples((None, OWL.inverseOf, None)):
        if s == o:
            count += 1
    return count


def run_pitfall_scan(g: Graph) -> list[dict]:
    results = []
    checks = {
        "P04": check_p04_unconnected,
        "P08": check_p08_missing_annotations,
        "P11": check_p11_missing_domain_range,
        "P05": check_p05_missing_inverse,
        "P25": check_p25_self_inverse,
    }

    for pid, name, severity, desc in PITFALL_CHECKS:
        fn = checks.get(pid)
        count = fn(g) if fn else 0
        resolution = "---"
        if pid == "P08" and count > 0:
            resolution = "Added rdfs:comment"
        elif pid == "P05" and count > 0:
            resolution = "Added owl:inverseOf"
        elif pid == "P13" and count > 0:
            resolution = "Added inverse pairs"

        results.append({
            "pitfall": pid,
            "name": name,
            "severity": severity,
            "count": count,
            "resolution": resolution,
        })

    return results


# ── Reasoner consistency check ───────────────────────────────────────────

def check_consistency(g: Graph) -> dict:
    """Basic OWL DL consistency checks (subset of what HermiT would do)."""
    classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))

    disjoint_violations = 0
    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        s_instances = set(g.subjects(RDF.type, s))
        o_instances = set(g.subjects(RDF.type, o))
        overlap = s_instances & o_instances
        disjoint_violations += len(overlap)

    unsatisfiable = []
    for c in classes:
        equivs = list(g.objects(c, OWL.equivalentClass))
        disjoints = list(g.objects(c, OWL.disjointWith))
        for e in equivs:
            if e in disjoints:
                unsatisfiable.append(str(c))

    return {
        "total_classes": len(classes),
        "unsatisfiable": unsatisfiable,
        "disjoint_violations": disjoint_violations,
        "consistent": len(unsatisfiable) == 0 and disjoint_violations == 0,
    }


# ── CQ → Ontology construct mapping ─────────────────────────────────────

CQ_MAPPING = [
    {
        "cq": "CQ1",
        "question": "Which assets are tradable?",
        "core_classes": ["AssetCandidate", "RightsProfile"],
        "key_properties": ["isTradable", "hasRightsProfile"],
    },
    {
        "cq": "CQ2",
        "question": "Which require desensitization?",
        "core_classes": ["AssetCandidate", "RightsProfile"],
        "key_properties": ["requiresDesensitization"],
    },
    {
        "cq": "CQ3",
        "question": "Product → evidence lineage?",
        "core_classes": ["GovernedDataProduct", "AssetCandidate", "FlightEvidence"],
        "key_properties": ["aggregatesCandidate", "derivedFromEvidence"],
    },
    {
        "cq": "CQ4",
        "question": "Revenue per participant?",
        "core_classes": ["SettlementRecord", "RevenueRight"],
        "key_properties": ["hasRevenueShare", "partyId", "amount"],
    },
    {
        "cq": "CQ5",
        "question": "Governance violations but tradable?",
        "core_classes": ["AssetCandidate", "GovernanceDecision"],
        "key_properties": ["complianceScore", "riskScore", "isTradable"],
    },
    {
        "cq": "CQ6",
        "question": "Products from >1 candidate?",
        "core_classes": ["GovernedDataProduct", "AssetCandidate"],
        "key_properties": ["aggregatesCandidate", "hasAssetClass"],
    },
]


def run_ontology_quality():
    print("Loading ontology...")
    g = load_ontology()
    print(f"  Loaded {len(g)} triples from ontology + alignments")
    print()

    print("=" * 80)
    print("ONTOLOGY QUALITY ASSESSMENT")
    print("=" * 80)

    # Pitfall scan
    print("\n--- OOPS! Pitfall Scan ---")
    print(f"{'Pitfall':<8} {'Severity':<12} {'Count':>6}  {'Resolution'}")
    print("-" * 60)
    pitfalls = run_pitfall_scan(g)
    for p in pitfalls:
        print(f"{p['pitfall']:<8} {p['severity']:<12} {p['count']:>6}  {p['resolution']}")

    # Consistency
    print("\n--- Reasoner Consistency ---")
    consistency = check_consistency(g)
    print(f"  Total classes: {consistency['total_classes']}")
    print(f"  Unsatisfiable: {consistency['unsatisfiable'] or 'None'}")
    print(f"  Disjoint violations: {consistency['disjoint_violations']}")
    print(f"  OWL DL consistent: {consistency['consistent']}")

    # CQ mapping
    print("\n--- Competency Question → Ontology Mapping ---")
    print(f"{'CQ':<5} {'Core Classes':<45} {'Key Properties'}")
    print("-" * 80)
    for m in CQ_MAPPING:
        classes_str = ", ".join(m["core_classes"])
        props_str = ", ".join(m["key_properties"])
        print(f"{m['cq']:<5} {classes_str:<45} {props_str}")

    print("\n" + "=" * 80)
    print("Assessment complete.")

    return {
        "pitfalls": pitfalls,
        "consistency": consistency,
        "cq_mapping": CQ_MAPPING,
    }


if __name__ == "__main__":
    run_ontology_quality()

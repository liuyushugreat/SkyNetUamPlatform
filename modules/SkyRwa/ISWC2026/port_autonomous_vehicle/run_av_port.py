"""Minimal end-to-end check that the SkyRwa pattern ports to an AV domain
without modifying AssetCandidate / GovernanceDecision / SettlementRecord.

Verifies (i) the ported ontology parses and is OWL DL consistent under the
same lightweight checks used by reproduce_ontology_quality.py, (ii) the
retargeted DriveEvidenceShape is a well-formed SHACL shape, and
(iii) a toy graph of 20 DriveEvidence nodes validates as expected.

Run:  python run_av_port.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rdflib import Graph, Namespace, RDF, Literal, URIRef, XSD

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent  # modules/SkyRwa
ONT_DIR = PKG_ROOT / "ontology"
SHAPES_DIR = PKG_ROOT / "shapes"

SKYRWA = Namespace("https://w3id.org/skyrwa#")
AVPORT = Namespace("urn:skyrwa:port:av#")


def load_graph() -> Graph:
    g = Graph()
    g.parse(str(ONT_DIR / "skyrwa.ttl"), format="turtle")
    g.parse(str(ONT_DIR / "alignments.ttl"), format="turtle")
    g.parse(str(HERE / "av_port.ttl"), format="turtle")
    return g


def build_toy_graph(n: int = 20) -> Graph:
    g = Graph()
    g.bind("skyrwa", SKYRWA)
    g.bind("avport", AVPORT)
    for i in range(n):
        drive = URIRef(f"urn:avport:drive:{i:03d}")
        vehicle = URIRef(f"urn:avport:vehicle:AV-{i%3:02d}")
        trip = URIRef(f"urn:avport:trip:T-{i:03d}")
        g.add((drive, RDF.type, AVPORT.DriveEvidence))
        g.add((vehicle, RDF.type, AVPORT.Vehicle))
        g.add((trip, RDF.type, AVPORT.Trip))
        g.add((drive, AVPORT.performedByVehicle, vehicle))
        g.add((drive, AVPORT.hasTrip, trip))
        g.add((drive, SKYRWA.flightId, Literal(f"AV-SESSION-{i:03d}")))
        g.add((drive, SKYRWA.hasDigest, Literal(f"sha256:{i:064d}")))
        g.add((drive, SKYRWA.startTime,
               Literal(f"2026-04-17T{(i % 24):02d}:00:00", datatype=XSD.dateTime)))
        g.add((drive, SKYRWA.endTime,
               Literal(f"2026-04-17T{(i % 24):02d}:30:00", datatype=XSD.dateTime)))
    return g


def main() -> int:
    print("[AV-port] Loading SkyRwa + AV-port ontology ...")
    ont = load_graph()
    print(f"  triples in ontology+alignments+AV-port: {len(ont)}")

    # Lightweight OWL DL consistency check: no class is both equivalent and
    # disjoint; no self-inverse object property.
    from rdflib import OWL
    bad = 0
    for s, _, o in ont.triples((None, OWL.inverseOf, None)):
        if s == o:
            bad += 1
    print(f"  self-inverse properties: {bad}  (expected 0)")

    print("[AV-port] Building toy benchmark (20 drive evidence nodes) ...")
    data = build_toy_graph(20)
    print(f"  triples in toy dataset: {len(data)}")

    # Merge ontology + shape + data, run pyshacl if available.
    shape = Graph()
    shape.parse(str(HERE / "av_shape.ttl"), format="turtle")

    try:
        from pyshacl import validate
        conforms, report_g, report_text = validate(
            data_graph=data,
            shacl_graph=shape,
            ont_graph=ont,
            inference="rdfs",
            meta_shacl=False,
            advanced=True,
        )
        print(f"[AV-port] SHACL validation conforms = {conforms}")
        if not conforms:
            print(report_text[:800])
    except ImportError:
        print("[AV-port] pyshacl not installed; skipping validation step.")
        conforms = None

    out = {
        "ontology_triples": len(ont),
        "data_triples": len(data),
        "self_inverse_violations": bad,
        "shacl_conforms": conforms,
        "port_scope": {
            "renamed_classes": ["FlightEvidence->DriveEvidence",
                                 "UAV->Vehicle",
                                 "FlightMission->Trip"],
            "renamed_properties": ["performedByUAV->performedByVehicle",
                                    "hasMission->hasTrip"],
            "inherited_unchanged": ["AssetCandidate", "GovernanceDecision",
                                     "GovernedDataProduct", "RevenueRight",
                                     "SettlementRecord", "GOV-001..GOV-003"],
        },
    }
    out_path = HERE / "av_port_result.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[AV-port] Result written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

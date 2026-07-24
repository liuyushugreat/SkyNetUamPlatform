"""Regenerate ontology serializations and HTML documentation.

Outputs (all relative to modules/SkyRwa/):
  ontology/skyrwa.owl          RDF/XML serialization of skyrwa.ttl
  ontology/skyrwa.jsonld       JSON-LD serialization of skyrwa.ttl
  ontology/skyrwa-merged.ttl   skyrwa.ttl + alignments.ttl in one graph
  ontology/skyrwa-merged.owl   RDF/XML of the merged graph
  docs/ontology/index.html     pyLODE documentation of the merged graph

The HTML documentation is rendered against the persistent w3id namespace
(https://w3id.org/skyrwa#), because docs/ontology/index.html is the redirect
target of https://w3id.org/skyrwa for HTML requests. Until migrate_namespace.py
has been applied, the source files still use https://w3id.org/skyrwa#, so this
script rewrites the namespace in memory before rendering (pyLODE also cannot
build fragment IDs for URN-scheme IRIs). After the migration this rewrite is a
no-op.

Usage (from modules/SkyRwa/):
    python docs/generate_ontology_docs.py

Requires: rdflib, pylode>=3  (pip install rdflib pylode)
"""
from pathlib import Path

from rdflib import Graph
from pylode import OntPub

MODULE_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_DIR = MODULE_ROOT / "ontology"
DOCS_DIR = MODULE_ROOT / "docs" / "ontology"

URN_NS = "https://w3id.org/skyrwa#"
W3ID_NS = "https://w3id.org/skyrwa#"


def main() -> None:
    core = ONTOLOGY_DIR / "skyrwa.ttl"
    alignments = ONTOLOGY_DIR / "alignments.ttl"

    g = Graph()
    g.parse(core)
    g.serialize(ONTOLOGY_DIR / "skyrwa.owl", format="pretty-xml")
    g.serialize(ONTOLOGY_DIR / "skyrwa.jsonld", format="json-ld", indent=2)
    print(f"core ontology: {len(g)} triples -> skyrwa.owl, skyrwa.jsonld")

    merged = Graph()
    merged.parse(core)
    merged.parse(alignments)
    merged_ttl = ONTOLOGY_DIR / "skyrwa-merged.ttl"
    merged.serialize(merged_ttl, format="turtle")
    merged.serialize(ONTOLOGY_DIR / "skyrwa-merged.owl", format="pretty-xml")
    print(f"merged graph:  {len(merged)} triples -> skyrwa-merged.ttl/.owl")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    w3id_ttl = DOCS_DIR / "skyrwa-w3id.ttl"
    w3id_ttl.write_text(
        merged_ttl.read_text(encoding="utf-8").replace(URN_NS, W3ID_NS),
        encoding="utf-8",
    )
    html_out = DOCS_DIR / "index.html"
    OntPub(w3id_ttl).make_html(destination=html_out)
    print(f"documentation: {html_out} (namespace rendered as {W3ID_NS})")


if __name__ == "__main__":
    main()

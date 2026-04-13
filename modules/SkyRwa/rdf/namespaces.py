"""Namespace declarations shared across the RDF layer."""

from rdflib import Namespace, Graph

SKYRWA = Namespace("urn:skyrwa:ontology#")
SKYRWA_INST = Namespace("urn:skyrwa:")

PROV = Namespace("http://www.w3.org/ns/prov#")
DCAT = Namespace("http://www.w3.org/ns/dcat#")
ODRL = Namespace("http://www.w3.org/ns/odrl/2/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
SCHEMA = Namespace("http://schema.org/")


def bind_namespaces(g: Graph) -> Graph:
    """Bind common prefixes to a graph for pretty serialization."""
    g.bind("skyrwa", SKYRWA)
    g.bind("inst", SKYRWA_INST)
    g.bind("prov", PROV)
    g.bind("dcat", DCAT)
    g.bind("odrl", ODRL)
    g.bind("dcterms", DCTERMS)
    g.bind("schema", SCHEMA)
    return g

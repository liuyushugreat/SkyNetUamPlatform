"""Product catalogue: registry of governed data products.

Provides lookup, listing, and RDF export of all governed products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from ..rdf.namespaces import SKYRWA, SKYRWA_INST, bind_namespaces
from .product_builder import GovernedProduct


@dataclass
class CatalogueEntry:
    product_id: str
    category: str
    source_count: int
    tradable: bool
    suggested_value: float
    lineage: str


class ProductCatalogue:
    """In-memory catalogue of governed data products."""

    def __init__(self) -> None:
        self._products: Dict[str, GovernedProduct] = {}

    def register(self, product: GovernedProduct) -> None:
        self._products[product.product_id] = product

    def get(self, product_id: str) -> Optional[GovernedProduct]:
        return self._products.get(product_id)

    def list_entries(self) -> List[CatalogueEntry]:
        return [
            CatalogueEntry(
                product_id=p.product_id,
                category=p.product_category.value,
                source_count=len(p.source_asset_ids),
                tradable=p.tradable,
                suggested_value=p.suggested_value,
                lineage=p.lineage_note,
            )
            for p in self._products.values()
        ]

    def to_graph(self) -> Graph:
        """Export the catalogue as an RDF graph."""
        g = Graph()
        bind_namespaces(g)
        for p in self._products.values():
            subj = SKYRWA_INST[f"product:{p.product_id}"]
            g.add((subj, RDF.type, SKYRWA.GovernedDataProduct))
            g.add((subj, SKYRWA.hasAssetClass, Literal(p.product_category.value)))
            g.add((subj, SKYRWA.isTradable, Literal(p.tradable, datatype=XSD.boolean)))
            g.add((subj, SKYRWA.estimatedValue, Literal(p.suggested_value, datatype=XSD.float)))
            for aid in p.source_asset_ids:
                candidate_iri = SKYRWA_INST[f"asset:{aid}"]
                g.add((subj, SKYRWA.aggregatesCandidate, candidate_iri))
            if p.rights_summary:
                g.add((subj, SKYRWA.requiresDesensitization,
                       Literal(p.rights_summary.desensitization_required, datatype=XSD.boolean)))
        return g

    def __len__(self) -> int:
        return len(self._products)

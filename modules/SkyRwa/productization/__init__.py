"""Multi-flight aggregation and data product lifecycle."""

from .aggregator import CandidateAggregator
from .product_builder import ProductBuilder, GovernedProduct
from .catalogue import ProductCatalogue, CatalogueEntry

__all__ = [
    "CandidateAggregator",
    "ProductBuilder",
    "GovernedProduct",
    "ProductCatalogue",
    "CatalogueEntry",
]

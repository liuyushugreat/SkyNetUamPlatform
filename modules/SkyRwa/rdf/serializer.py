"""Convenience functions for serializing domain objects to RDF formats."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from pydantic import BaseModel
from rdflib import Graph

from .mapper import SkyRwaMapper
from .namespaces import bind_namespaces

from ..models.evidence import FlightEvidencePackage
from ..models.asset_unit import FlightAssetUnit
from ..models.rights import RightsProfile
from ..models.settlement import SettlementRule, RevenueLog, SettlementRecord
from ..models.valuation import ValuationResultV2

_DISPATCH = {
    FlightEvidencePackage: "map_evidence",
    FlightAssetUnit: "map_asset_unit",
    RightsProfile: "map_rights_profile",
    SettlementRule: "map_settlement_rule",
    RevenueLog: "map_revenue_log",
    SettlementRecord: "map_settlement_record",
    ValuationResultV2: "map_valuation",
}


def to_graph(obj: BaseModel, graph: Graph | None = None) -> Graph:
    """Map a single domain object into an RDF graph."""
    mapper = SkyRwaMapper(graph)
    cls = type(obj)
    method_name = _DISPATCH.get(cls)
    if method_name is None:
        raise TypeError(f"No RDF mapping for {cls.__name__}")
    getattr(mapper, method_name)(obj)
    return mapper.graph


def to_turtle(obj: BaseModel) -> str:
    """Serialize a domain object as Turtle."""
    g = to_graph(obj)
    return g.serialize(format="turtle")


def to_jsonld(obj: BaseModel) -> str:
    """Serialize a domain object as JSON-LD."""
    g = to_graph(obj)
    return g.serialize(format="json-ld")


def save_graph(graph: Graph, path: Union[str, Path], fmt: str = "turtle") -> None:
    """Persist a graph to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(p), format=fmt)

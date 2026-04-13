"""In-memory RDF graph store for the SkyRwa knowledge graph.

Wraps a single :class:`rdflib.Graph` and provides convenience helpers
for loading ontology files, executing SPARQL queries, and persisting /
loading the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from rdflib import Graph

from .namespaces import bind_namespaces


class GraphStore:
    """Lightweight graph store backed by rdflib."""

    def __init__(self, graph: Optional[Graph] = None):
        self.graph = graph if graph is not None else Graph()
        bind_namespaces(self.graph)

    # ── Loading ─────────────────────────────────────────────────────────

    def load_file(self, path: Union[str, Path], fmt: str = "turtle") -> None:
        self.graph.parse(str(path), format=fmt)

    def load_ontology(self, ontology_dir: Union[str, Path] | None = None) -> None:
        """Load all ``.ttl`` files from the ontology directory."""
        if ontology_dir is None:
            ontology_dir = Path(__file__).resolve().parent.parent / "ontology"
        ontology_dir = Path(ontology_dir)
        for ttl in sorted(ontology_dir.glob("*.ttl")):
            self.graph.parse(str(ttl), format="turtle")

    def load_shapes(self, shapes_dir: Union[str, Path] | None = None) -> None:
        """Load all SHACL shapes from the shapes directory."""
        if shapes_dir is None:
            shapes_dir = Path(__file__).resolve().parent.parent / "shapes"
        shapes_dir = Path(shapes_dir)
        for ttl in sorted(shapes_dir.glob("*.ttl")):
            self.graph.parse(str(ttl), format="turtle")

    # ── Querying ────────────────────────────────────────────────────────

    def query(self, sparql: str) -> List[dict]:
        """Execute a SPARQL SELECT and return rows as dicts."""
        result = self.graph.query(sparql)
        rows: List[dict] = []
        vars_ = [str(v) for v in result.vars] if result.vars else []
        for row in result:
            rows.append({v: row[i] for i, v in enumerate(vars_)})
        return rows

    def query_file(self, path: Union[str, Path]) -> List[dict]:
        sparql = Path(path).read_text(encoding="utf-8")
        return self.query(sparql)

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: Union[str, Path], fmt: str = "turtle") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(p), format=fmt)

    def __len__(self) -> int:
        return len(self.graph)

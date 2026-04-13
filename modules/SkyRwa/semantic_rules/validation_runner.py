"""SHACL validation runner.

Loads a data graph and SHACL shapes, then runs pyshacl validation and
returns structured results suitable for tests and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from rdflib import Graph
import pyshacl


@dataclass
class ValidationViolation:
    focus_node: str
    path: str
    message: str
    severity: str = "Violation"


@dataclass
class ValidationReport:
    conforms: bool
    violations: List[ValidationViolation] = field(default_factory=list)
    raw_text: str = ""


class ShaclValidator:
    """Validate an RDF graph against SkyRwa SHACL shapes."""

    def __init__(self, shapes_dir: Optional[Union[str, Path]] = None):
        self.shapes_graph = Graph()
        shapes_dir = Path(shapes_dir) if shapes_dir else (
            Path(__file__).resolve().parent.parent / "shapes"
        )
        for ttl in sorted(shapes_dir.glob("*.ttl")):
            self.shapes_graph.parse(str(ttl), format="turtle")

    def validate(self, data_graph: Graph) -> ValidationReport:
        """Run SHACL validation and return a structured report."""
        conforms, results_graph, results_text = pyshacl.validate(
            data_graph,
            shacl_graph=self.shapes_graph,
            inference="none",
            abort_on_first=False,
        )
        violations: List[ValidationViolation] = []
        if not conforms:
            from rdflib.namespace import SH
            SH_NS = SH
            for result in results_graph.subjects(
                predicate=None, object=SH_NS.ValidationResult
            ):
                focus = str(results_graph.value(result, SH_NS.focusNode) or "")
                path = str(results_graph.value(result, SH_NS.resultPath) or "")
                msg = str(results_graph.value(result, SH_NS.resultMessage) or "")
                sev = str(results_graph.value(result, SH_NS.resultSeverity) or "Violation")
                violations.append(ValidationViolation(
                    focus_node=focus, path=path, message=msg, severity=sev,
                ))
        return ValidationReport(
            conforms=conforms,
            violations=violations,
            raw_text=results_text,
        )

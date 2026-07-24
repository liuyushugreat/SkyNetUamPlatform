"""SHACL validation runners.

Loads a data graph and SHACL shapes, then runs validation and returns
structured results suitable for tests and experiments.

Two interchangeable engines are provided behind the same interface:

* :class:`ShaclValidator`  -- pySHACL (Python; SHACL Core + SHACL-SPARQL
  via ``advanced=True``).
* :class:`RudofValidator`  -- rudof (Rust, via the ``pyrudof`` bindings;
  SHACL Core only -- ``sh:sparql`` constraints are not evaluated by
  rudof as of pyrudof 0.3.x and are silently skipped).
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
            advanced=True,
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


class RudofValidator:
    """Validate an RDF graph against SkyRwa SHACL shapes using rudof.

    Same ``validate(data_graph) -> ValidationReport`` interface as
    :class:`ShaclValidator`. rudof evaluates SHACL Core constraints only;
    the one ``sh:sparql`` constraint in AssetCandidateShape is skipped.

    Requires ``pip install pyrudof``.
    """

    def __init__(self, shapes_dir: Optional[Union[str, Path]] = None):
        from pyrudof import Rudof, RudofConfig  # deferred: optional dependency

        shapes_dir = Path(shapes_dir) if shapes_dir else (
            Path(__file__).resolve().parent.parent / "shapes"
        )
        self.shapes_text = "\n".join(
            ttl.read_text(encoding="utf-8")
            for ttl in sorted(shapes_dir.glob("*.ttl"))
        )
        self._rudof = Rudof(RudofConfig())
        self._rudof.read_shacl(self.shapes_text)

    def validate(self, data_graph: Graph) -> ValidationReport:
        """Serialize the graph to Turtle and validate it with rudof."""
        return self.validate_turtle(data_graph.serialize(format="turtle"))

    def validate_turtle(self, turtle_data: str) -> ValidationReport:
        """Validate Turtle text directly (lets callers time serialization
        separately from engine execution)."""
        from pyrudof import ResultShaclValidationFormat

        self._rudof.reset_data()
        self._rudof.reset_validation_results()
        self._rudof.read_data(turtle_data)
        self._rudof.validate_shacl()
        report_ttl = self._rudof.serialize_shacl_validation_results(
            ResultShaclValidationFormat.Turtle
        )

        from rdflib.namespace import SH
        report_graph = Graph()
        report_graph.parse(data=report_ttl, format="turtle")
        conforms_val = next(report_graph.objects(None, SH.conforms), None)
        conforms = str(conforms_val).lower() == "true"
        violations: List[ValidationViolation] = []
        if not conforms:
            for result in report_graph.subjects(None, SH.ValidationResult):
                focus = str(report_graph.value(result, SH.focusNode) or "")
                path = str(report_graph.value(result, SH.resultPath) or "")
                msg = str(report_graph.value(result, SH.resultMessage) or "")
                sev = str(report_graph.value(result, SH.resultSeverity)
                          or "Violation")
                violations.append(ValidationViolation(
                    focus_node=focus, path=path, message=msg, severity=sev,
                ))
        return ValidationReport(
            conforms=conforms,
            violations=violations,
            raw_text=report_ttl,
        )

"""Coverage matrix for the SkyRwa benchmark.

Prints a human-readable table that maps each scenario to the violation
types it exercises, the governance paths it covers, and the lifecycle
transitions it tests.

Run standalone::

    python -m SkyRwa.benchmark_generator.coverage_matrix
"""

from __future__ import annotations

from .scenario_spec import SCENARIO_SPECS

# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

VIOLATION_DIMS: list[tuple[str, str]] = [
    ("struct_violation",     "Structural Violation"),
    ("threshold_violation",  "Threshold Violation"),
    ("conditional_violation","Conditional Violation"),
    ("emergent_violation",   "Emergent Violation"),
]

GOVERNANCE_DIMS: list[tuple[str, str]] = [
    ("desensitization",  "Desensitisation Gate"),
    ("direct_promotion", "Direct Promotion"),
    ("standard_gov",     "Standard Governance"),
    ("rights_conflict",  "Rights Conflict"),
    ("mission_failure",  "Mission Failure"),
    ("quality_failure",  "Quality Failure"),
]

LIFECYCLE_DIMS: list[tuple[str, str]] = [
    ("promotion",  "Promotion"),
    ("settlement", "Settlement"),
    ("rejection",  "Rejection"),
    ("partial",    "Partial Pass"),
]

# ---------------------------------------------------------------------------
# Per-scenario coverage declaration
# ---------------------------------------------------------------------------
# Each entry is keyed by scenario tag and holds sets of active dimension keys.
# ---------------------------------------------------------------------------

_COVERAGE: dict[str, dict[str, set[str]]] = {
    "clean_route_survey": {
        "violations":  set(),
        "governance":  {"direct_promotion"},
        "lifecycle":   {"promotion", "settlement"},
    },
    "night_flight": {
        "violations":  {"struct_violation"},
        "governance":  {"standard_gov"},
        "lifecycle":   {"rejection"},
    },
    "weather_disturbance": {
        "violations":  {"struct_violation"},
        "governance":  {"standard_gov"},
        "lifecycle":   {"rejection"},
    },
    "near_nfz": {
        "violations":  {"threshold_violation", "conditional_violation"},
        "governance":  {"standard_gov"},
        "lifecycle":   {"promotion", "rejection", "partial"},
    },
    "anomaly_maintenance": {
        "violations":  {"emergent_violation"},
        "governance":  {"standard_gov"},
        "lifecycle":   {"rejection"},
    },
    "emergency_logistics": {
        "violations":  {"threshold_violation", "conditional_violation"},
        "governance":  {"mission_failure"},
        "lifecycle":   {"promotion", "rejection", "partial"},
    },
    "low_quality": {
        "violations":  {"struct_violation", "conditional_violation"},
        "governance":  {"quality_failure"},
        "lifecycle":   {"rejection"},
    },
    "rights_conflict": {
        "violations":  set(),
        "governance":  {"rights_conflict", "desensitization"},
        "lifecycle":   {"promotion", "settlement"},
    },
    "beyond_vlos": {
        "violations":  {"threshold_violation", "conditional_violation"},
        "governance":  {"standard_gov"},
        "lifecycle":   {"promotion", "rejection", "partial"},
    },
    "urban_corridor": {
        "violations":  {"threshold_violation", "emergent_violation"},
        "governance":  {"standard_gov", "mission_failure"},
        "lifecycle":   {"promotion", "rejection", "partial"},
    },
}


def _mark(covered: bool) -> str:
    return "Y" if covered else "-"


def _col_width(header: str, min_width: int = 5) -> int:
    return max(len(header), min_width)


def print_coverage_matrix() -> None:
    all_dims = (
        [k for k, _ in VIOLATION_DIMS]
        + [k for k, _ in GOVERNANCE_DIMS]
        + [k for k, _ in LIFECYCLE_DIMS]
    )
    all_labels = (
        [lbl for _, lbl in VIOLATION_DIMS]
        + [lbl for _, lbl in GOVERNANCE_DIMS]
        + [lbl for _, lbl in LIFECYCLE_DIMS]
    )

    tag_col = max(len(s["tag"]) for s in SCENARIO_SPECS) + 2
    col_widths = [max(len(lbl), 5) for lbl in all_labels]

    sep = "+" + "-" * tag_col + "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = (
        "| " + "Scenario".ljust(tag_col - 2) + " |"
        + "".join(f" {lbl.center(w)} |" for lbl, w in zip(all_labels, col_widths))
    )

    print(sep)
    print(header_row)
    print(sep.replace("-", "="))

    for spec in SCENARIO_SPECS:
        tag = spec["tag"]
        coverage = _COVERAGE.get(tag, {})
        viol_set = coverage.get("violations", set())
        gov_set  = coverage.get("governance", set())
        life_set = coverage.get("lifecycle", set())
        combined = viol_set | gov_set | life_set

        row = "| " + tag.ljust(tag_col - 2) + " |"
        for dim, w in zip(all_dims, col_widths):
            row += f" {_mark(dim in combined).center(w)} |"
        print(row)
        print(sep)


def get_coverage_dict() -> dict[str, dict[str, list[str]]]:
    """Return coverage as a serialisable dict (useful for JSON export)."""
    return {
        tag: {
            "violations":  sorted(data.get("violations", set())),
            "governance":  sorted(data.get("governance", set())),
            "lifecycle":   sorted(data.get("lifecycle", set())),
        }
        for tag, data in _COVERAGE.items()
    }


if __name__ == "__main__":
    print("\nSkyRwa Benchmark Coverage Matrix\n")
    print_coverage_matrix()
    print(
        "\nDimensions: Structural = schema/field violations; "
        "Threshold = numeric exceedance; Conditional = index-gated injection; "
        "Emergent = arises from pipeline scoring.\n"
    )

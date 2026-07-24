"""Submit the SkyRwa ontology to FOOPS! (Ontology Pitfall Scanner for FAIR)
and store the raw JSON response, including the overall FAIRness score.

FOOPS! API: https://w3id.org/foops/api  (Garijo, Corcho, Poveda-Villalon)
Two endpoints are supported; file upload is preferred so the assessment runs
against the exact local ontology state:
  POST /assessOntologyFile   multipart file upload
  POST /assessOntology       JSON {"ontologyUri": "..."} (fallback)

Usage (from modules/SkyRwa/):
    python evaluation/run_foops_evaluation.py

Output:
    evaluation/foops_report.json   raw response from the FOOPS! service
"""
import json
from pathlib import Path

import requests

MODULE_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_TTL = MODULE_ROOT / "ontology" / "skyrwa-merged.ttl"
REPORT_PATH = Path(__file__).resolve().parent / "foops_report.json"

FOOPS_BASE = "https://foops.linkeddata.es"
FALLBACK_URI = "https://w3id.org/skyrwa"


def assess_file() -> requests.Response:
    with ONTOLOGY_TTL.open("rb") as fh:
        return requests.post(
            f"{FOOPS_BASE}/assessOntologyFile",
            files={"file": (ONTOLOGY_TTL.name, fh, "text/turtle")},
            headers={"accept": "application/json;charset=UTF-8"},
            timeout=600,
        )


def assess_uri() -> requests.Response:
    return requests.post(
        f"{FOOPS_BASE}/assessOntology",
        json={"ontologyUri": FALLBACK_URI},
        headers={"accept": "application/json;charset=UTF-8",
                 "Content-Type": "application/json;charset=UTF-8"},
        timeout=600,
    )


def main() -> None:
    print(f"Submitting {ONTOLOGY_TTL.name} to FOOPS! (file upload) ...")
    resp = assess_file()
    if resp.status_code != 200:
        print(f"file upload returned HTTP {resp.status_code}; "
              f"falling back to ontologyUri = {FALLBACK_URI}")
        resp = assess_uri()
    resp.raise_for_status()

    report = resp.json()
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    score = report.get("overall_score")
    checks = report.get("checks", [])
    passed = sum(1 for c in checks
                 if c.get("total_passed_tests") == c.get("total_tests_run"))
    print(f"Saved raw FOOPS! response to {REPORT_PATH}")
    print(f"Overall FAIRness score: {score}")
    print(f"Checks fully passed: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()

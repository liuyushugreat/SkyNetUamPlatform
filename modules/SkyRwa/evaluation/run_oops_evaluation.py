"""Submit the SkyRwa ontology to the OOPS! (OntOlogy Pitfall Scanner!) web
service and store the raw XML response.

OOPS! REST API: https://oops.linkeddata.es/rest  (Poveda-Villalon et al.)
The service accepts an RDF/XML ontology serialization inside an XML request
envelope and returns the detected pitfalls as XML.

Usage (from modules/SkyRwa/):
    python evaluation/run_oops_evaluation.py

Output:
    evaluation/oops_report.xml   raw response from the OOPS! service

Render a human-readable summary afterwards with:
    python evaluation/render_oops_report.py
"""
from pathlib import Path

import requests

MODULE_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_RDFXML = MODULE_ROOT / "ontology" / "skyrwa-merged.owl"
REPORT_PATH = Path(__file__).resolve().parent / "oops_report.xml"

OOPS_ENDPOINT = "https://oops.linkeddata.es/rest"

REQUEST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<OOPSRequest>
  <OntologyUrl></OntologyUrl>
  <OntologyContent><![CDATA[
{content}
]]></OntologyContent>
  <Pitfalls></Pitfalls>
  <OutputFormat>XML</OutputFormat>
</OOPSRequest>
"""


def main() -> None:
    content = ONTOLOGY_RDFXML.read_text(encoding="utf-8")
    body = REQUEST_TEMPLATE.format(content=content)
    print(f"Submitting {ONTOLOGY_RDFXML.name} "
          f"({len(content)} chars) to {OOPS_ENDPOINT} ...")
    resp = requests.post(
        OOPS_ENDPOINT,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/xml"},
        timeout=300,
    )
    resp.raise_for_status()
    REPORT_PATH.write_bytes(resp.content)
    print(f"Saved raw OOPS! response to {REPORT_PATH} "
          f"({len(resp.content)} bytes)")


if __name__ == "__main__":
    main()

"""Render evaluation/oops_report.xml (raw OOPS! web-service response) as a
human-readable summary table.

Usage (from modules/SkyRwa/):
    python evaluation/render_oops_report.py            # print to stdout
    python evaluation/render_oops_report.py --markdown # also write oops_summary.md
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parent / "oops_report.xml"
SUMMARY_PATH = Path(__file__).resolve().parent / "oops_summary.md"

OOPS_NS = "{http://www.oeg-upm.net/oops}"


def parse_report(path: Path):
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    pitfalls = []
    for pf in root.iter(f"{OOPS_NS}Pitfall"):
        code = pf.findtext(f"{OOPS_NS}Code", default="?")
        name = pf.findtext(f"{OOPS_NS}Name", default="?")
        importance = pf.findtext(f"{OOPS_NS}Importance", default="?")
        # NumberAffectedElements is occasionally emitted outside the Pitfall
        # element by the service (e.g. for ontology-wide pitfalls like P10);
        # fall back to counting AffectedElement entries.
        num = pf.findtext(f"{OOPS_NS}NumberAffectedElements")
        affected = [el.text for el in pf.iter(f"{OOPS_NS}AffectedElement")]
        count = int(num) if num is not None else (len(affected) or 1)
        external_only = bool(affected) and all(
            not a.startswith("urn:skyrwa:") and "w3id.org/skyrwa" not in a
            for a in affected
        )
        pitfalls.append({
            "code": code,
            "name": name,
            "importance": importance,
            "count": count,
            "affected": affected,
            "external_only": external_only,
        })
    return pitfalls


def render(pitfalls) -> str:
    order = {"Critical": 0, "Important": 1, "Minor": 2}
    pitfalls = sorted(pitfalls, key=lambda p: (order.get(p["importance"], 9),
                                               p["code"]))
    lines = [
        "# OOPS! evaluation summary",
        "",
        f"Source: raw response in `oops_report.xml` "
        f"(OOPS! web service, https://oops.linkeddata.es/).",
        "",
        "| Code | Pitfall | Importance | Affected | Scope |",
        "|------|---------|------------|----------|-------|",
    ]
    for p in pitfalls:
        scope = ("external vocabulary terms only" if p["external_only"]
                 else "ontology-wide" if not p["affected"]
                 else "skyrwa terms")
        lines.append(f"| {p['code']} | {p['name']} | {p['importance']} | "
                     f"{p['count']} | {scope} |")
    crit = sum(1 for p in pitfalls if p["importance"] == "Critical")
    lines += [
        "",
        f"Critical pitfalls: {crit}. "
        f"Total pitfall categories: {len(pitfalls)}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true",
                        help=f"also write {SUMMARY_PATH.name}")
    args = parser.parse_args()

    summary = render(parse_report(REPORT_PATH))
    print(summary)
    if args.markdown:
        SUMMARY_PATH.write_text(summary, encoding="utf-8")
        print(f"written: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

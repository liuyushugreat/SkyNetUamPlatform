"""One-shot namespace migration: urn:skyrwa:ontology#  ->  https://w3id.org/skyrwa#

Run AFTER the w3id.org registration is approved (see w3id-submission.md).

Usage (from modules/SkyRwa/):
    python migrate_namespace.py           # dry run: list files and match counts
    python migrate_namespace.py --apply   # rewrite files in place

The script walks this module's directory tree and rewrites every occurrence in
*.ttl, *.py and *.rq files. Generated artifacts (JSON-LD/OWL serializations,
sample graphs) are also plain text and are rewritten too; regenerate them
afterwards anyway with docs/generate_ontology_docs.py for consistency.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OLD_NS = "urn:skyrwa:ontology#"
NEW_NS = "https://w3id.org/skyrwa#"

EXTENSIONS = {".ttl", ".py", ".rq", ".owl", ".jsonld"}
SKIP_DIRS = {"__pycache__", ".git", "node_modules"}
SELF = Path(__file__).resolve()


def find_targets(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue
        if path.resolve() == SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        count = text.count(OLD_NS)
        if count:
            yield path, text, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="rewrite files in place (default: dry run)")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent),
                        help="directory tree to migrate (default: this module)")
    args = parser.parse_args()

    root = Path(args.root)
    total_files = 0
    total_hits = 0
    for path, text, count in find_targets(root):
        total_files += 1
        total_hits += count
        rel = path.relative_to(root)
        print(f"{count:5d}  {rel}")
        if args.apply:
            path.write_text(text.replace(OLD_NS, NEW_NS), encoding="utf-8")

    mode = "REWRITTEN" if args.apply else "DRY RUN (use --apply to rewrite)"
    print(f"\n{total_files} file(s), {total_hits} occurrence(s) of {OLD_NS}")
    print(mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())

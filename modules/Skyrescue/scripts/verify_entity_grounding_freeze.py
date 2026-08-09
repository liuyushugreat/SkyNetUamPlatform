#!/usr/bin/env python3
"""Verify the frozen entity-grounding artifacts before confirmatory scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    module_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=module_root / "configs" / "entity_grounding_freeze_v1.0.0.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = []
    valid = True
    for artifact in manifest["artifacts"]:
        path = module_root / artifact["path"]
        actual = file_sha256(path) if path.is_file() else None
        matched = actual == artifact["sha256"]
        valid = valid and matched
        checks.append({
            "path": artifact["path"],
            "expected_sha256": artifact["sha256"],
            "actual_sha256": actual,
            "matched": matched,
        })

    print(json.dumps({
        "freeze": manifest["artifact"],
        "valid": valid,
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()

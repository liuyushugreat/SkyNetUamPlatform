"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to the YAML config.")
    p.add_argument("--out", default="outputs",
                   help="Output directory (created if missing).")
    return p


def ensure_outputs(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

"""Shared helpers used by the experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


def default_config_path() -> Path:
    return MODULE_ROOT / "configs" / "default.yaml"


def scaling_config_path() -> Path:
    return MODULE_ROOT / "configs" / "scaling.yaml"


def ablation_config_path() -> Path:
    return MODULE_ROOT / "configs" / "ablation.yaml"

"""Test fixtures shared across the SkyShield pytest suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from skyshield.config import load_config


ROOT = Path(__file__).parent.parent


@pytest.fixture()
def default_cfg():
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture()
def field_sorties_path():
    return ROOT / "data" / "field_sorties.json"

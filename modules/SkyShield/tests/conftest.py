"""Pytest configuration: ensure the in-tree ``skyshield`` package is importable
without installation, and expose a deterministic config fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyshield.config import SkyShieldConfig  # noqa: E402


@pytest.fixture(scope="session")
def default_config() -> SkyShieldConfig:
    return SkyShieldConfig.load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def module_root() -> Path:
    return ROOT

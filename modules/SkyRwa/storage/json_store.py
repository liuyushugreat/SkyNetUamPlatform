"""Simple JSON file persistence for asset units and ledger entries.

This is a lightweight, file-based store suitable for prototyping and local
testing.  In production it would be replaced by a database-backed
repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from ..models.asset_unit import FlightAssetUnit


class JsonStore:
    """Read / write :class:`FlightAssetUnit` objects as JSON files.

    Each asset unit is persisted as ``<asset_unit_id>.json`` inside *base_dir*.
    """

    def __init__(self, base_dir: str | Path = "skyrwa_store"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, unit: FlightAssetUnit) -> Path:
        path = self._path_for(unit.asset_unit_id)
        data = unit.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def load(self, asset_unit_id: str) -> Optional[FlightAssetUnit]:
        path = self._path_for(asset_unit_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return FlightAssetUnit(**data)

    def list_ids(self) -> List[str]:
        return [
            p.stem for p in self.base_dir.glob("*.json")
        ]

    def list_all(self) -> List[FlightAssetUnit]:
        units: List[FlightAssetUnit] = []
        for uid in self.list_ids():
            u = self.load(uid)
            if u is not None:
                units.append(u)
        return units

    def delete(self, asset_unit_id: str) -> bool:
        path = self._path_for(asset_unit_id)
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Ledger persistence helpers
    # ------------------------------------------------------------------

    def save_ledger(self, entries: List[dict], filename: str = "ledger.json") -> Path:
        path = self.base_dir / filename
        path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
        return path

    def load_ledger(self, filename: str = "ledger.json") -> List[dict]:
        path = self.base_dir / filename
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path_for(self, asset_unit_id: str) -> Path:
        return self.base_dir / f"{asset_unit_id}.json"

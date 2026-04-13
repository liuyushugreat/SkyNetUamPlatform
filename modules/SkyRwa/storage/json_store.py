"""Simple JSON file persistence for asset units, ledger and settlement records.

This is a lightweight, file-based store suitable for prototyping and local
testing.  In production it would be replaced by a database-backed repository.

Persistence layout inside *base_dir*::

    <base_dir>/
        <asset_unit_id>.json      -- one file per asset unit
        ledger.json               -- all RevenueLog entries
        settlements.json          -- all SettlementRecord snapshots

TODO(db): replace with SQLAlchemy / async DB adapter for production.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..models.asset_unit import FlightAssetUnit


class JsonStore:
    """Read / write :class:`FlightAssetUnit`, revenue-log and settlement data.

    Each asset unit is persisted as ``<asset_unit_id>.json`` inside *base_dir*.

    Raises
    ------
    OSError
        If *base_dir* cannot be created.
    """

    def __init__(self, base_dir: str | Path = "skyrwa_store"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Asset Unit CRUD
    # ------------------------------------------------------------------

    def save(self, unit: FlightAssetUnit) -> Path:
        """Persist an asset unit to ``<asset_unit_id>.json``."""
        path = self._path_for(unit.asset_unit_id)
        data = unit.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def load(self, asset_unit_id: str) -> Optional[FlightAssetUnit]:
        """Load an asset unit by id.  Returns ``None`` if not found."""
        path = self._path_for(asset_unit_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return FlightAssetUnit(**data)

    def list_ids(self) -> List[str]:
        return [
            p.stem
            for p in sorted(self.base_dir.glob("*.json"))
            if p.stem not in ("ledger", "settlements")
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
    # Ledger (RevenueLog) persistence
    # ------------------------------------------------------------------

    def save_ledger(self, entries: List[dict], filename: str = "ledger.json") -> Path:
        """Write all revenue-log entries to a single JSON file."""
        path = self.base_dir / filename
        path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
        return path

    def load_ledger(self, filename: str = "ledger.json") -> List[dict]:
        path = self.base_dir / filename
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Settlement record persistence
    # ------------------------------------------------------------------

    def save_settlements(
        self, records: List[dict], filename: str = "settlements.json",
    ) -> Path:
        """Write all settlement records to a single JSON file."""
        path = self.base_dir / filename
        path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        return path

    def load_settlements(self, filename: str = "settlements.json") -> List[dict]:
        path = self.base_dir / filename
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path_for(self, asset_unit_id: str) -> Path:
        return self.base_dir / f"{asset_unit_id}.json"

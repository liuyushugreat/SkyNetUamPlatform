"""In-memory + JSON-persisted revenue ledger.

Each call to :meth:`record_usage` creates a :class:`RevenueLog`, applies the
:class:`RevenueSplitter`, and appends the entry to the asset unit's log and
the ledger's internal list.  The ledger can be serialised to / loaded from a
JSON file via the :class:`~SkyRwa.storage.json_store.JsonStore` helper.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetStatus, SettlementStatus, UsageType
from ..models.settlement import RevenueLog, SettlementRule, SplitEntry
from .splitter import RevenueSplitter


class Ledger:
    """Append-only revenue ledger backed by an in-memory list."""

    def __init__(self) -> None:
        self._entries: List[RevenueLog] = []
        self._splitter = RevenueSplitter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_usage(
        self,
        unit: FlightAssetUnit,
        usage_type: UsageType,
        consumer: str,
        gross_amount: float,
        metadata: Optional[Dict[str, str]] = None,
    ) -> RevenueLog:
        """
        Record a revenue-generating event against *unit*.

        If the unit has a :class:`SettlementRule`, the gross amount is
        automatically split among participants.
        """
        rule = unit.settlement_rule or SettlementRule()
        splits = self._splitter.split(gross_amount, rule)

        log = RevenueLog(
            asset_unit_id=unit.asset_unit_id,
            usage_type=usage_type,
            consumer=consumer,
            gross_amount=gross_amount,
            split_detail=splits,
            settlement_status=SettlementStatus.PENDING,
            metadata=metadata or {},
        )
        self._entries.append(log)
        unit.revenue_log.append(log)
        if unit.status == AssetStatus.VALUATED:
            unit.status = AssetStatus.SETTLEMENT_READY
        unit.updated_at = datetime.utcnow()
        return log

    def settle(self, usage_id: str) -> bool:
        """Mark a pending entry as ``SETTLED``."""
        for entry in self._entries:
            if entry.usage_id == usage_id:
                entry.settlement_status = SettlementStatus.SETTLED
                return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def entries(self) -> List[RevenueLog]:
        return list(self._entries)

    def for_asset(self, asset_unit_id: str) -> List[RevenueLog]:
        return [e for e in self._entries if e.asset_unit_id == asset_unit_id]

    def total_revenue(self, asset_unit_id: Optional[str] = None) -> float:
        subset = self.for_asset(asset_unit_id) if asset_unit_id else self._entries
        return sum(e.gross_amount for e in subset)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dicts(self) -> List[dict]:
        return [e.model_dump(mode="json") for e in self._entries]

    @classmethod
    def from_dicts(cls, data: List[dict]) -> "Ledger":
        ledger = cls()
        ledger._entries = [RevenueLog(**d) for d in data]
        return ledger

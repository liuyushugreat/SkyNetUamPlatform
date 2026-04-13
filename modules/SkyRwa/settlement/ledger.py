"""In-memory + JSON-persisted revenue ledger.

Each call to :meth:`record_usage` creates a :class:`RevenueLog`, applies the
:class:`RevenueSplitter`, and appends the entry to the asset unit's log and
the ledger's internal list.

Calling :meth:`settle` finalises one or more revenue-log entries and produces
a :class:`SettlementRecord` snapshot that can be persisted separately.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Dict, List, Optional

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetStatus, SettlementStatus, UsageType
from ..models.settlement import (
    RevenueLog,
    SettlementRecord,
    SettlementRule,
    SplitEntry,
)
from .splitter import RevenueSplitter


class Ledger:
    """Append-only revenue ledger with settlement-record generation."""

    def __init__(self) -> None:
        self._entries: List[RevenueLog] = []
        self._settlements: List[SettlementRecord] = []
        self._splitter = RevenueSplitter()

    # ------------------------------------------------------------------
    # Revenue recording
    # ------------------------------------------------------------------

    def record_usage(
        self,
        unit: FlightAssetUnit,
        usage_type: UsageType,
        consumer: str,
        gross_amount: float,
        metadata: Optional[Dict[str, str]] = None,
    ) -> RevenueLog:
        """Record a revenue-generating event against *unit*.

        Raises
        ------
        ValueError
            If *gross_amount* is negative.
        """
        if gross_amount < 0:
            raise ValueError(f"gross_amount must be >= 0, got {gross_amount}")

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
        unit.updated_at = datetime.now(UTC)
        return log

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, usage_id: str) -> Optional[SettlementRecord]:
        """Mark a pending entry as SETTLED and return a :class:`SettlementRecord`.

        Returns ``None`` if the *usage_id* is not found.
        """
        for entry in self._entries:
            if entry.usage_id == usage_id:
                entry.settlement_status = SettlementStatus.SETTLED
                record = SettlementRecord(
                    asset_unit_id=entry.asset_unit_id,
                    settled_usage_ids=[entry.usage_id],
                    total_gross=entry.gross_amount,
                    participant_totals=list(entry.split_detail),
                    currency=entry.metadata.get("currency", "USD"),
                )
                self._settlements.append(record)
                return record
        return None

    def settle_all(self, asset_unit_id: str) -> Optional[SettlementRecord]:
        """Settle all pending entries for an asset and return a combined record."""
        pending = [
            e for e in self._entries
            if e.asset_unit_id == asset_unit_id
            and e.settlement_status == SettlementStatus.PENDING
        ]
        if not pending:
            return None

        for e in pending:
            e.settlement_status = SettlementStatus.SETTLED

        total_gross = sum(e.gross_amount for e in pending)
        aggregated = self._aggregate_splits(pending)

        record = SettlementRecord(
            asset_unit_id=asset_unit_id,
            settled_usage_ids=[e.usage_id for e in pending],
            total_gross=total_gross,
            participant_totals=aggregated,
        )
        self._settlements.append(record)
        return record

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def entries(self) -> List[RevenueLog]:
        return list(self._entries)

    @property
    def settlements(self) -> List[SettlementRecord]:
        return list(self._settlements)

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

    def settlements_to_dicts(self) -> List[dict]:
        return [s.model_dump(mode="json") for s in self._settlements]

    @classmethod
    def from_dicts(cls, data: List[dict]) -> "Ledger":
        ledger = cls()
        ledger._entries = [RevenueLog(**d) for d in data]
        return ledger

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_splits(entries: List[RevenueLog]) -> List[SplitEntry]:
        """Sum per-participant amounts across multiple revenue logs."""
        totals: Dict[str, SplitEntry] = {}
        for e in entries:
            for s in e.split_detail:
                key = s.party_id
                if key in totals:
                    totals[key].amount = round(totals[key].amount + s.amount, 6)
                else:
                    totals[key] = SplitEntry(
                        party_id=s.party_id,
                        role=s.role,
                        share_pct=s.share_pct,
                        amount=s.amount,
                    )
        return list(totals.values())

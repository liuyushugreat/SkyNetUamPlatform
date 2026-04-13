"""Tests for revenue splitting and ledger operations."""

from __future__ import annotations

from SkyRwa.models.enums import SettlementStatus, UsageType
from SkyRwa.models.settlement import SettlementRule, SplitEntry
from SkyRwa.settlement.ledger import Ledger
from SkyRwa.settlement.splitter import RevenueSplitter


class TestRevenueSplitter:
    def test_split_amounts_sum_to_gross(self, default_settlement_rule):
        splitter = RevenueSplitter()
        splits = splitter.split(100.0, default_settlement_rule)
        total = sum(s.amount for s in splits)
        assert abs(total - 100.0) < 0.01

    def test_split_normalises_percentages(self):
        rule = SettlementRule(
            participants=[
                SplitEntry(party_id="a", role="x", share_pct=60),
                SplitEntry(party_id="b", role="y", share_pct=40),
            ],
        )
        splitter = RevenueSplitter()
        splits = splitter.split(200.0, rule)
        assert abs(splits[0].amount - 120.0) < 0.01
        assert abs(splits[1].amount - 80.0) < 0.01

    def test_empty_participants(self):
        rule = SettlementRule(participants=[])
        splitter = RevenueSplitter()
        splits = splitter.split(50.0, rule)
        assert splits == []

    def test_split_preserves_party_info(self, default_settlement_rule):
        splitter = RevenueSplitter()
        splits = splitter.split(10.0, default_settlement_rule)
        roles = {s.role for s in splits}
        assert "platform" in roles
        assert "operator" in roles
        assert "data_processor" in roles


class TestLedger:
    def test_record_usage_creates_log(self, governed_unit, default_settlement_rule):
        governed_unit.settlement_rule = default_settlement_rule
        ledger = Ledger()
        log = ledger.record_usage(
            governed_unit,
            usage_type=UsageType.API_CALL,
            consumer="test-consumer",
            gross_amount=10.0,
        )
        assert log.gross_amount == 10.0
        assert log.consumer == "test-consumer"
        assert log.settlement_status == SettlementStatus.PENDING

    def test_log_appended_to_unit(self, governed_unit, default_settlement_rule):
        governed_unit.settlement_rule = default_settlement_rule
        ledger = Ledger()
        ledger.record_usage(governed_unit, UsageType.API_CALL, "c", 5.0)
        ledger.record_usage(governed_unit, UsageType.TRAINING_USE, "d", 15.0)
        assert len(governed_unit.revenue_log) == 2

    def test_total_revenue(self, governed_unit, default_settlement_rule):
        governed_unit.settlement_rule = default_settlement_rule
        ledger = Ledger()
        ledger.record_usage(governed_unit, UsageType.API_CALL, "c", 5.0)
        ledger.record_usage(governed_unit, UsageType.API_CALL, "d", 15.0)
        assert ledger.total_revenue() == 20.0

    def test_settle_marks_entry(self, governed_unit, default_settlement_rule):
        governed_unit.settlement_rule = default_settlement_rule
        ledger = Ledger()
        log = ledger.record_usage(governed_unit, UsageType.API_CALL, "c", 5.0)
        assert ledger.settle(log.usage_id)
        assert log.settlement_status == SettlementStatus.SETTLED

    def test_serialisation_roundtrip(self, governed_unit, default_settlement_rule):
        governed_unit.settlement_rule = default_settlement_rule
        ledger = Ledger()
        ledger.record_usage(governed_unit, UsageType.API_CALL, "c", 10.0)
        data = ledger.to_dicts()
        restored = Ledger.from_dicts(data)
        assert len(restored.entries) == 1
        assert restored.entries[0].gross_amount == 10.0

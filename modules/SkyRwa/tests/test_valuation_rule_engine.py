"""Tests for the rule-based valuation engine."""

from __future__ import annotations

from SkyRwa.models.enums import AssetStatus
from SkyRwa.valuation.rule_engine import RuleBasedValuationEngine


class TestRuleBasedValuationEngine:
    def test_evaluate_returns_result(self, governed_unit):
        engine = RuleBasedValuationEngine()
        result = engine.evaluate(governed_unit)
        assert result.asset_unit_id == governed_unit.asset_unit_id
        assert result.engine_id == "rule_based"

    def test_status_advanced_to_valuated(self, governed_unit):
        engine = RuleBasedValuationEngine()
        engine.evaluate(governed_unit)
        assert governed_unit.status == AssetStatus.VALUATED

    def test_estimated_value_positive(self, governed_unit):
        engine = RuleBasedValuationEngine()
        result = engine.evaluate(governed_unit)
        assert result.estimated_value > 0

    def test_quality_score_dimensions(self, governed_unit):
        engine = RuleBasedValuationEngine()
        result = engine.evaluate(governed_unit)
        qs = result.quality_score
        assert 0 <= qs.completeness <= 1
        assert 0 <= qs.temporal_continuity <= 1
        assert 0 <= qs.sensor_reliability <= 1
        assert 0 <= qs.event_richness <= 1
        assert 0 <= qs.compliance_degree <= 1
        assert 0 <= qs.overall <= 1

    def test_value_score_dimensions(self, governed_unit):
        engine = RuleBasedValuationEngine()
        result = engine.evaluate(governed_unit)
        vs = result.value_score
        assert 0 <= vs.scarcity <= 1
        assert 0 <= vs.scenario_relevance <= 1
        assert 0 <= vs.reuse_potential <= 1
        assert 0 <= vs.timeliness <= 1
        assert 0 <= vs.overall <= 1

    def test_breakdown_contains_weights(self, governed_unit):
        engine = RuleBasedValuationEngine()
        result = engine.evaluate(governed_unit)
        assert "quality_weight" in result.breakdown
        assert "value_weight" in result.breakdown
        assert "combined_factor" in result.breakdown

    def test_custom_base_price(self, governed_unit):
        engine_low = RuleBasedValuationEngine(base_price=10.0)
        engine_high = RuleBasedValuationEngine(base_price=1000.0)
        r_low = engine_low.evaluate(governed_unit)
        governed_unit.status = AssetStatus.GOVERNED  # reset for re-eval
        r_high = engine_high.evaluate(governed_unit)
        assert r_high.estimated_value > r_low.estimated_value

    def test_unit_quality_score_updated(self, governed_unit):
        engine = RuleBasedValuationEngine()
        result = engine.evaluate(governed_unit)
        assert governed_unit.data_quality_score == result.quality_score.overall

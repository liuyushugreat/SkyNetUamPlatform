"""Tests for multi-flight aggregation and productization."""

from datetime import UTC, datetime

import pytest

from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.models.evidence import FlightEvidencePackage
from SkyRwa.models.rights import RightsProfile, RevenueParticipant
from SkyRwa.models.valuation import ValuationResultV2
from SkyRwa.models.enums import AssetClass, UsageLevel
from SkyRwa.productization.aggregator import CandidateAggregator
from SkyRwa.productization.product_builder import ProductBuilder, GovernedProduct
from SkyRwa.productization.catalogue import ProductCatalogue
from SkyRwa.valuation.product_valuation import ProductValuationEngine


def _make_candidate(i: int, asset_class=AssetClass.ROUTE_OPTIMIZATION_SAMPLE,
                    quality=0.8, compliance=0.9, tradable=True):
    ev = FlightEvidencePackage(
        flight_id=f"FLT-AGG-{i:03d}", uav_id=f"UAV-A{i%3+1}",
        start_time=datetime(2026, 1, 1, 8 + i, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, 8 + i, 30, tzinfo=UTC),
        digest_hash=f"digest{i}",
    )
    return FlightAssetUnit(
        flight_id=f"FLT-AGG-{i:03d}", uav_id=f"UAV-A{i%3+1}",
        evidence=ev,
        asset_class=asset_class,
        compliance_score=compliance,
        data_quality_score=quality,
        rights_profile=RightsProfile(
            owner="OP", tradable=tradable,
            permitted_uses=[UsageLevel.LICENSED_EXTERNAL],
            revenue_split=[
                RevenueParticipant(party_id="OP", role="operator", share_pct=60.0),
                RevenueParticipant(party_id="PLAT", role="platform", share_pct=40.0),
            ],
        ),
        valuation_result=ValuationResultV2(
            asset_unit_id=f"AU-{i}", estimated_value=50.0 + i * 5,
        ),
    )


class TestCandidateAggregator:
    def test_groups_by_class(self):
        candidates = [_make_candidate(i) for i in range(5)]
        agg = CandidateAggregator(min_count=3)
        groups = agg.group(candidates)
        assert AssetClass.ROUTE_OPTIMIZATION_SAMPLE in groups
        assert groups[AssetClass.ROUTE_OPTIMIZATION_SAMPLE].count == 5

    def test_filters_low_quality(self):
        candidates = [_make_candidate(i, quality=0.2) for i in range(5)]
        agg = CandidateAggregator(min_count=3)
        groups = agg.group(candidates)
        assert len(groups) == 0

    def test_filters_non_tradable(self):
        candidates = [_make_candidate(i, tradable=False) for i in range(5)]
        agg = CandidateAggregator(min_count=3)
        groups = agg.group(candidates)
        assert len(groups) == 0

    def test_min_count_threshold(self):
        candidates = [_make_candidate(i) for i in range(2)]
        agg = CandidateAggregator(min_count=3)
        groups = agg.group(candidates)
        assert len(groups) == 0


class TestProductBuilder:
    def test_builds_product(self):
        candidates = [_make_candidate(i) for i in range(4)]
        agg = CandidateAggregator(min_count=3)
        groups = agg.group(candidates)
        builder = ProductBuilder()
        for cls, group in groups.items():
            product = builder.build(group)
            assert isinstance(product, GovernedProduct)
            assert len(product.source_asset_ids) == 4
            assert product.tradable is True
            assert product.suggested_value > 0

    def test_too_few_candidates_raises(self):
        from SkyRwa.productization.aggregator import AggregationGroup
        group = AggregationGroup(
            asset_class=AssetClass.RISK_DATASET,
            candidates=[_make_candidate(0)],
        )
        builder = ProductBuilder()
        with pytest.raises(ValueError):
            builder.build(group)

    def test_merged_participants(self):
        candidates = [_make_candidate(i) for i in range(3)]
        agg = CandidateAggregator(min_count=2)
        groups = agg.group(candidates)
        builder = ProductBuilder()
        for cls, group in groups.items():
            product = builder.build(group)
            assert len(product.rights_summary.revenue_split) == 2


class TestProductCatalogue:
    def test_register_and_list(self):
        product = GovernedProduct(
            product_category=AssetClass.RISK_DATASET,
            source_asset_ids=["A1", "A2", "A3"],
        )
        catalogue = ProductCatalogue()
        catalogue.register(product)
        entries = catalogue.list_entries()
        assert len(entries) == 1
        assert entries[0].source_count == 3

    def test_to_graph(self):
        product = GovernedProduct(
            product_category=AssetClass.WEATHER_OPERATION_SAMPLE,
            source_asset_ids=["A1", "A2", "A3"],
            tradable=True,
            suggested_value=500.0,
        )
        catalogue = ProductCatalogue()
        catalogue.register(product)
        g = catalogue.to_graph()
        assert len(g) >= 3


class TestProductValuation:
    def test_valuates_product(self):
        candidates = [_make_candidate(i) for i in range(5)]
        agg = CandidateAggregator(min_count=3)
        groups = agg.group(candidates)
        builder = ProductBuilder()
        engine = ProductValuationEngine()
        for cls, group in groups.items():
            product = builder.build(group)
            explanation = engine.valuate(product)
            assert explanation.final_value > 0
            assert len(explanation.factors) == 4
            assert explanation.promotion_readiness == "ready"

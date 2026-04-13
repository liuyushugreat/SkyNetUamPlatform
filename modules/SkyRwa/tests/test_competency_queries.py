"""Tests for SPARQL competency queries against a populated graph."""

from datetime import UTC, datetime

import pytest
from rdflib import Graph

from SkyRwa.rdf.mapper import SkyRwaMapper
from SkyRwa.rdf.graph_store import GraphStore
from SkyRwa.rdf.namespaces import bind_namespaces
from SkyRwa.models.evidence import FlightEvidencePackage
from SkyRwa.models.asset_unit import FlightAssetUnit
from SkyRwa.models.rights import RightsProfile, RevenueParticipant
from SkyRwa.models.valuation import ValuationResultV2
from SkyRwa.models.settlement import RevenueLog, SettlementRecord, SplitEntry
from SkyRwa.models.enums import AssetClass, UsageLevel, UsageType


def _build_graph():
    g = Graph()
    bind_namespaces(g)
    mapper = SkyRwaMapper(g)

    for i in range(4):
        ev = FlightEvidencePackage(
            flight_id=f"FLT-CQ-{i:03d}", uav_id=f"UAV-CQ{i%2+1}",
            start_time=datetime(2026, 2, 1, 8 + i, tzinfo=UTC),
            end_time=datetime(2026, 2, 1, 8 + i, 30, tzinfo=UTC),
            digest_hash=f"cqdigest{i}",
        )
        unit = FlightAssetUnit(
            flight_id=f"FLT-CQ-{i:03d}", uav_id=f"UAV-CQ{i%2+1}",
            evidence=ev,
            compliance_score=0.9 if i < 3 else 0.3,
            risk_score=0.1 if i < 3 else 0.9,
            data_quality_score=0.8 if i < 3 else 0.2,
            rights_profile=RightsProfile(
                owner="OP-CQ",
                tradable=i < 3,
                desensitization_required=i == 1,
                permitted_uses=[UsageLevel.LICENSED_EXTERNAL] if i < 3 else [UsageLevel.INTERNAL_ONLY],
                revenue_split=[
                    RevenueParticipant(party_id="OP-CQ", role="operator", share_pct=60.0),
                    RevenueParticipant(party_id="PLAT", role="platform", share_pct=40.0),
                ],
            ),
            valuation_result=ValuationResultV2(
                asset_unit_id=f"AU-CQ-{i}", estimated_value=80.0 if i < 3 else 10.0,
                breakdown={"completeness": 0.9 if i < 3 else 0.2},
            ),
        )
        mapper.map_asset_unit(unit)

    rec = SettlementRecord(
        asset_unit_id="AU-CQ-0",
        total_gross=100.0,
        participant_totals=[
            SplitEntry(party_id="OP-CQ", role="operator", share_pct=60.0, amount=60.0),
            SplitEntry(party_id="PLAT", role="platform", share_pct=40.0, amount=40.0),
        ],
    )
    mapper.map_settlement_record(rec)
    return g


@pytest.fixture
def store():
    g = _build_graph()
    s = GraphStore(g)
    s.load_ontology()
    return s


class TestCompetencyQueries:
    def test_cq1_tradable_assets(self, store):
        from pathlib import Path
        qdir = Path(__file__).resolve().parent.parent / "queries" / "competency"
        rows = store.query_file(qdir / "cq_01_tradable_assets.rq")
        assert len(rows) == 3

    def test_cq2_desensitization(self, store):
        from pathlib import Path
        qdir = Path(__file__).resolve().parent.parent / "queries" / "competency"
        rows = store.query_file(qdir / "cq_02_assets_requiring_desensitization.rq")
        assert len(rows) >= 1

    def test_cq4_revenue_by_participant(self, store):
        from pathlib import Path
        qdir = Path(__file__).resolve().parent.parent / "queries" / "competency"
        rows = store.query_file(qdir / "cq_04_revenue_by_participant.rq")
        assert len(rows) >= 2
        total = sum(float(r["totalRevenue"]) for r in rows)
        assert total == pytest.approx(100.0, abs=0.1)


class TestAnalyticalQueries:
    def test_q1_promotable_flights(self, store):
        from pathlib import Path
        qdir = Path(__file__).resolve().parent.parent / "queries" / "analytical"
        rows = store.query_file(qdir / "q_01_promotable_flights.rq")
        assert len(rows) >= 1

    def test_q2_governance_failures(self, store):
        from pathlib import Path
        qdir = Path(__file__).resolve().parent.parent / "queries" / "analytical"
        rows = store.query_file(qdir / "q_02_governance_failures.rq")
        assert len(rows) >= 1

    def test_q4_lineage(self, store):
        from pathlib import Path
        qdir = Path(__file__).resolve().parent.parent / "queries" / "analytical"
        rows = store.query_file(qdir / "q_04_asset_lineage.rq")
        assert len(rows) >= 1
